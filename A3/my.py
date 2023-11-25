import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(num_points, dimensions):
    #return np.random.randint(1, 6, size=(num_points, dimensions))
    return np.random.uniform(0, 1, size=(num_points, dimensions))

def compute_and_find_distances(query_points, dataset):
    ratios_l1 = []
    ratios_l2 = []
    ratios_linf = []

    for i in range(query_points.shape[0]):
        current_query_point = query_points[i]
        
        # Compute distances for the current query point
        diff = dataset - current_query_point[np.newaxis, :]
        
        # Compute L1 distances
        l1_distances = np.linalg.norm(diff, ord=1, axis=1)
        nearest_index_l1 = np.argmin(l1_distances)
        farthest_index_l1 = np.argmax(l1_distances)
        ratio_l1 = l1_distances[farthest_index_l1] / l1_distances[nearest_index_l1]
        ratios_l1.append(ratio_l1)
       
        # Compute L2 distances
        l2_distances = np.linalg.norm(diff, ord=2, axis=1)
        nearest_index_l2 = np.argmin(l2_distances)
        farthest_index_l2 = np.argmax(l2_distances)
        ratio_l2 = l2_distances[farthest_index_l2] / l2_distances[nearest_index_l2]
        ratios_l2.append(ratio_l2)
        
        # Compute Linfinity distances
        linf_distances = np.linalg.norm(diff, ord=np.inf, axis=1)
        nearest_index_linf = np.argmin(linf_distances)
        farthest_index_linf = np.argmax(linf_distances)
        ratio_linf = linf_distances[farthest_index_linf] / linf_distances[nearest_index_linf]
        ratios_linf.append(ratio_linf)
      

    return round(np.mean(ratios_l1),5), round(np.mean(ratios_l2),5), round(np.mean(ratios_linf),5)

num_points = 1000000
num_queries = 100
dimensions_list = [1,2,4,8,16,32,64]
ratios_l1 = []
ratios_l2 = []
ratios_linf = []

if __name__ == "__main__":
    
    for dimensions in dimensions_list:
        dataset = generate_dataset(num_points, dimensions)
        query_indices = np.random.choice(num_points, num_queries, replace=False)
        query_points = dataset[query_indices]

        # Ensure that the query points are not included in the dataset
        mask = np.all(dataset[:, None, :] == query_points, axis=2).any(axis=1)
        dataset = dataset[~mask]
        
        ratio_l1, ratio_l2, ratio_linf = compute_and_find_distances(query_points, dataset)
        # print(dimensions)

        ratios_l1.append(ratio_l1)
        ratios_l2.append(ratio_l2)
        ratios_linf.append(ratio_linf)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions_list, ratios_l1, marker='o', label='L1')
    plt.plot(dimensions_list, ratios_l2, marker='o', label='L2')
    plt.plot(dimensions_list, ratios_linf, marker='o', label='L∞')
    plt.xlabel('Dimensions')
    plt.ylabel('Average Ratio (Farthest/Nearest)')
    plt.title('Behavior of Uniform Distribution in High-Dimensional Spaces')
    plt.yscale('log', basey =2)
    plt.legend()
    plt.savefig('Q1.png')
    plt.close()
