import torch
from torch_geometric.data import Data

def gen_distance_matrix(tsp_coordinates):
    n_nodes = len(tsp_coordinates)
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e9
    return distances
    
def gen_pyg_data(tsp_coordinates, k_sparse):
    n_nodes = len(tsp_coordinates)
    distances = gen_distance_matrix(tsp_coordinates)
    topk_values, topk_indices = torch.topk(distances, 
                                           k=k_sparse, 
                                           dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                repeats=k_sparse),
        torch.flatten(topk_indices)
        ])
    edge_attr = topk_values.reshape(-1, 1)
    pyg_data = Data(x=tsp_coordinates, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances

def load_val_dataset(n_node, k_sparse, device):
    val_list = []
    val_tensor = torch.load(f'data/tsp/valDataset-{n_node}.pt')
    for instance in val_tensor:
        instance = instance.to(device)
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse)
        val_list.append((data, distances))
    return val_list

def load_test_dataset(n_node, k_sparse, device):
    val_list = []
    val_tensor = torch.load(f'data/tsp/testDataset-{n_node}.pt') 
    for instance in val_tensor:
        instance = instance.to(device)
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse)
        val_list.append((data, distances))
    return val_list



def generate_and_save_fixed_dataset(n_node, steps_per_epoch, epochs, file_path):
    total_instances = steps_per_epoch * epochs 
    instances = []
    
    for _ in range(total_instances):
        instance = torch.rand(size=(n_node, 2), device='cuda:0')  
        instances.append(instance)
    torch.save({'instances': instances}, file_path)
    print(f"Fixed dataset saved to {file_path}")

# generate_and_save_fixed_dataset(500 , 512 ,20 , f'tsp/traindata{500}')



