import pathlib
import torch
from torch_geometric.data import Data
import pickle

def instance_gen(n, device):
    due_time_norm = torch.rand(size=(n,), device=device)  # [n,]
    due_time = due_time_norm * (n)
    weights = torch.rand(size=(n,), device=device)  # [n,]
    processing_time = torch.rand(size=(n,), device=device) # [n]
    
    x = torch.stack([due_time_norm, weights]).T # (n, 2)
    x_depot = torch.zeros(size=(1, 2), device=device)
    x = torch.cat([x_depot, x], dim=0)
    
    _edge_attr = torch.cat([torch.zeros(size=(1,), device=device), processing_time]) # (n+1,) 
    edge_attr = torch.repeat_interleave(_edge_attr, n+1).unsqueeze(-1) # attr of <i,j> is the processing time of j
    nodes = torch.arange(n+1, device=device)
    u = nodes.repeat(n+1)
    v = torch.repeat_interleave(nodes, n+1)
    edge_index = torch.stack([u,v])
    pyg_data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
    return pyg_data, due_time, weights, processing_time

def load_test_dataset(n_node, device):
    with open(f"data/smtwtp/testDataset-{n_node}.pkl", "rb") as f:
        loaded_list = pickle.load(f)
    for i in range(len(loaded_list)):
        for j in range(len(loaded_list[0])):
            loaded_list[i][j] = loaded_list[i][j].to(device)
    return loaded_list



def generate_and_save_datasets(problem_sizes, num_train, num_val, dataset_dir):
    torch.manual_seed(123456)  
    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)  
    
    for p_size in problem_sizes:
        train_instances = []
        val_instances = []
        
        # Generate training instances
        for _ in range(num_train):
            pyg_data, due_time, weights, processing_time = instance_gen(p_size, 'cpu')
            train_instances.append({
                'pyg_data': pyg_data,
                'due_time': due_time,
                'weights': weights,
                'processing_time': processing_time
            })
        
        # Generate validation instances
        for _ in range(num_val):
            pyg_data, due_time, weights, processing_time = instance_gen(p_size, 'cpu')
            val_instances.append({
                'pyg_data': pyg_data,
                'due_time': due_time,
                'weights': weights,
                'processing_time': processing_time
            })
        
    
        dataset = {
            'train_instances': train_instances,
            'val_instances': val_instances
        }
        with open(f'{dataset_dir}/train_val_datasets-{p_size}.pkl', 'wb') as f:
            pickle.dump(dataset, f)

        print(f"Dataset for problem size {p_size} saved (Train: {num_train}, Validation: {num_val})")


if __name__ == '__main__':
    problem_sizes = [100]  # Different problem sizes
    num_train = 20*512  # Number of training instances per problem size
    num_val = 30  # Number of validation instances per problem size
    dataset_dir = 'smtwtp'  # Directory to save datasets
    generate_and_save_datasets(problem_sizes, num_train, num_val, dataset_dir)