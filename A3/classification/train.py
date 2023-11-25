import argparse
import os
import gzip
import pandas as pd
import numpy as np
import torch

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Sequential, GCNConv, GATConv
from torch.nn import Linear, Softmax
from torch_geometric.nn import global_mean_pool
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Training a classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    args = parser.parse_args()
    print(f"Training a classification model. Output will be saved at {args.model_path}. Dataset will be loaded from {args.dataset_path}. Validation dataset will be loaded from {args.val_dataset_path}.")
    
    #parameters
    train_data_path = args.dataset_path #directory path
    val_data_path = args.val_dataset_path #directory path
    model_save_path = os.path.join(args.model_path, 'classification_model')

    encode_node = False
    encode_edge = False

    node_encoder = None
    edge_encoder = None

    node_emb_dim = 9
    edge_emb_dim = 3

    node_features = node_emb_dim if ((encode_node) and (node_encoder != None)) else 9
    edge_features = edge_emb_dim if ((encode_edge) and (edge_encoder != None)) else 3

    batch_size = 64
    n_epochs = 50
    hidden_channels = 15
    att_heads=1
    learning_rate = 0.01
    
    class GraphDataset(Dataset):
        def __init__(self, dataset_path, node_encoder=None, edge_encoder=None):
            super(GraphDataset, self).__init__()
            self.labels_path = os.path.join(dataset_path, 'graph_labels.csv.gz')
            self.n_nodes_path = os.path.join(dataset_path, 'num_nodes.csv.gz')
            self.node_features_path = os.path.join(dataset_path, 'node_features.csv.gz')
            self.n_edges_path = os.path.join(dataset_path, 'num_edges.csv.gz')
            self.edges_path = os.path.join(dataset_path, 'edges.csv.gz')
            self.edge_features_path = os.path.join(dataset_path, 'edge_features.csv.gz')

            #self.data, self.slices = self.process()
            self.data_list = []
            self.setup()

        def setup(self):

            with gzip.open(self.labels_path, 'rt') as file:
                labels_df = pd.read_csv(file, header=None)
            with gzip.open(self.n_nodes_path, 'rt') as file:
                n_nodes_df = pd.read_csv(file, header=None)
            with gzip.open(self.node_features_path, 'rt') as file:
                node_features_df = pd.read_csv(file, header=None)
            with gzip.open(self.n_edges_path, 'rt') as file:
                n_edges_df = pd.read_csv(file, header=None)
            with gzip.open(self.edges_path, 'rt') as file:
                edges_df = pd.read_csv(file, header=None)
            with gzip.open(self.edge_features_path, 'rt') as file:
                edge_features_df = pd.read_csv(file, header=None)

            n_graphs = len(labels_df)
            node_start = [0 for i in range(n_graphs)]
            for i in range(1, n_graphs):
                node_start[i] = node_start[i-1] + int(n_nodes_df.iloc[i-1, 0])

            edge_start = [0 for i in range(n_graphs)]
            for i in range(1, n_graphs):
                edge_start[i] = edge_start[i-1] + int(n_edges_df.iloc[i-1, 0])

            for ii in range(len(labels_df)):
                if ((labels_df.iloc[ii,0] != 0) and (labels_df.iloc[ii,0] != 1)):
                    continue

                label = torch.tensor([labels_df.iloc[ii, 0]], dtype=torch.long)

                n_nodes = n_nodes_df.iloc[ii, 0]
                node_start_index = node_start[ii]
                node_features = torch.tensor(node_features_df.iloc[node_start_index:node_start_index+n_nodes].values, dtype=torch.float)

                n_edges = n_edges_df.iloc[ii, 0]
                edge_start_index = edge_start[ii]
                edge_indices = torch.tensor(edges_df.iloc[edge_start_index:edge_start_index+n_edges].values, dtype=torch.long).t().contiguous()
                edge_features = torch.tensor(edge_features_df.iloc[edge_start_index:edge_start_index+n_edges].values, dtype=torch.float)

                data = Data(x=node_features, edge_index=edge_indices, edge_attr=edge_features, y=label)

                self.data_list.append(data)

        def len(self):
            return len(self.data_list)

        def get(self, idx):
            return self.data_list[idx]
          
            
    #Creating the dataset
    train_dataset = GraphDataset(dataset_path=train_data_path, node_encoder=node_encoder, edge_encoder=edge_encoder)
    val_dataset = GraphDataset(dataset_path=val_data_path, node_encoder=node_encoder, edge_encoder=edge_encoder)
    
    
    #Mini-batched loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    #Defining the model
    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            torch.manual_seed(12345)
            self.conv1 = GCNConv(train_dataset.num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.lin = Linear(hidden_channels, train_dataset.num_classes)

            self.conv1.reset_parameters()
            self.conv2.reset_parameters()
            self.conv3.reset_parameters()
            self.lin.reset_parameters()

        def forward(self, x, edge_index, batch):
            # 1. Obtain node embeddings
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)

            # 2. Readout layer
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

            # 3. Apply a final classifier
            #x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin(x)

            return x

    model = GCN(hidden_channels=hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    criterion = torch.nn.BCELoss()
    
    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
             data = data.to(device)
             out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
             m = Softmax(dim=1)
             soft_out = m(out)
             loss = criterion(soft_out, F.one_hot(data.y, num_classes=2).float())  # Compute the loss.
             loss.backward()  # Derive gradients.
             
             #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
             #min_grad_value = 0.000005
             #for param in model.parameters():
             #    if param.grad is not None:
             #        param.grad.data = torch.clamp(param.grad.data, min=min_grad_value)

             #for name, param in model.named_parameters():
             #   if param.grad is not None:
             #       print(f'{name} - Gradient Mean: {param.grad.mean().item()}, Gradient Std: {param.grad.std().item()}')

             optimizer.step()  # Update parameters based on gradients.
             optimizer.zero_grad()  # Clear gradients.

    def test(loader):
         model.eval()

         #correct = 0
         #for data in loader:  # Iterate in batches over the training/test dataset.
         #    out = model(data.x, data.edge_index, data.batch)
         #    pred = out.argmax(dim=1)  # Use the class with highest probability.
         #    correct += int((pred == data.y).sum())  # Check against ground-truth labels.
         #return correct / len(loader.dataset)  # Derive ratio of correct predictions.

         all_labels = []
         all_probs = []

         with torch.no_grad():
             for data in loader:
                 out = model(data.x, data.edge_index, data.batch)
                 probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()  # Assuming binary classification

                 all_labels.extend(data.y.cpu().numpy())
                 all_probs.extend(probs)

         roc_auc = roc_auc_score(all_labels, all_probs)
         return roc_auc

    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs):
        train()
        train_acc = test(train_loader)
        val_acc = test(val_loader)
        train_losses.append(train_acc)
        val_losses.append(val_acc)
        #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {val_acc:.4f}')
        
    torch.save(model.state_dict(), model_save_path)

if __name__=="__main__":
    main()
