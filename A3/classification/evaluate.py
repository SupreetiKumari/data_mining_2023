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
     
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tocsv(y_arr, *, task):
    r"""Writes the numpy array to a csv file.
    params:
        y_arr: np.ndarray. A vector of all the predictions. Classes for
        classification and the regression value predicted for regression.

        task: str. Must be either of "classification" or "regression".
        Must be a keyword argument.
    Outputs a file named "y_classification.csv" or "y_regression.csv" in
    the directory it is called from. Must only be run once. In case outputs
    are generated from batches, only call this output on all the predictions
    from all the batches collected in a single numpy array. This means it'll
    only be called once.

    This code ensures this by checking if the file already exists, and does
    not over-write the csv files. It just raises an error.

    Finally, do not shuffle the test dataset as then matching the outputs
    will not work.
    """
    import os
    import numpy as np
    import pandas as pd
    assert task in ["classification", "regression"], f"task must be either \"classification\" or \"regression\". Found: {task}"
    assert isinstance(y_arr, np.ndarray), f"y_arr must be a numpy array, found: {type(y_arr)}"
    assert len(y_arr.squeeze().shape) == 1, f"y_arr must be a vector. shape found: {y_arr.shape}"
    assert not os.path.isfile(f"y_{task}.csv"), f"File already exists. Ensure you are not calling this function multiple times (e.g. when looping over batches). Read the docstring. Found: y_{task}.csv"
    y_arr = y_arr.squeeze()
    df = pd.DataFrame(y_arr)
    df.to_csv(f"y_{task}.csv", index=False, header=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluating the classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()
    print(f"Evaluating the classification model. Model will be loaded from {args.model_path}. Test dataset will be loaded from {args.dataset_path}.")
    
    #parameters
    #train_data_path = args.dataset_path #directory path
    val_data_path = args.dataset_path #directory path
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
    #train_dataset = GraphDataset(dataset_path=train_data_path, node_encoder=node_encoder, edge_encoder=edge_encoder)
    val_dataset = GraphDataset(dataset_path=val_data_path, node_encoder=node_encoder, edge_encoder=edge_encoder)
    
    
    #Mini-batched loader
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    #Defining the model
    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            torch.manual_seed(12345)
            self.conv1 = GCNConv(9, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.lin = Linear(hidden_channels, 2)

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
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    all_ys = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(output, dim=1)[:, 1].cpu().numpy()
            ys_this_batch = probs.tolist()
            all_ys.extend(ys_this_batch)
    numpy_ys = np.asarray(all_ys)
    tocsv(numpy_ys, task="classification") # <- Called outside the loop. Called in the eval code.


if __name__=="__main__":
    main()
