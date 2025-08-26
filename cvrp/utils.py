import torch
from torch_geometric.data import Data

CAPACITY = 50
DEMAND_LOW = 1
DEMAND_HIGH = 9
DEPOT_COOR = [0.5, 0.5]

def gen_instance(n, device):
    locations = torch.rand(size=(n, 2), device=device)
    demands = torch.randint(low=DEMAND_LOW, high=DEMAND_HIGH+1, size=(n,), device=device)
    depot = torch.tensor([DEPOT_COOR], device=device)
    all_locations = torch.cat((depot, locations), dim=0)
    all_demands = torch.cat((torch.zeros((1,), device=device), demands))
    distances = gen_distance_matrix(all_locations)
    return all_demands, distances # (n+1), (n+1, n+1)

def gen_distance_matrix(tsp_coordinates):
    n_nodes = len(tsp_coordinates)
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e-10 # note here
    return distances

def gen_pyg_data(demands, distances, device):
    n = demands.size(0)
    nodes = torch.arange(n, device=device)
    u = nodes.repeat(n)
    v = torch.repeat_interleave(nodes, n)
    edge_index = torch.stack((u, v))
    edge_attr = distances.reshape(((n)**2, 1))
    x = demands
    pyg_data = Data(x=x.unsqueeze(1), edge_attr=edge_attr, edge_index=edge_index)
    return pyg_data

def load_test_dataset(problem_size, device):
    test_list = []
    dataset = torch.load(f'./data/cvrp/testDataset-{problem_size}.pt', map_location=device)
    for i in range(len(dataset)):
        test_list.append((dataset[i, 0, :], dataset[i, 1:, :]))
    return test_list

def generate_and_save_fixed_cvrp_datasets(n_customers, steps_per_epoch, max_epochs, test_size, file_path):

    # Total instances needed for training
    total_train_instances = steps_per_epoch * max_epochs
    train_instances = []
    
    for _ in range(total_train_instances):
        # Generate train CVRP instance using gen_instance
        demands, distances = gen_instance(n_customers, device='cpu')
        instance = {"demands": demands, "distances": distances}
        train_instances.append(instance)
    
    # Generate test instances
    test_instances = []
    for _ in range(test_size):
        # Generate test CVRP instance using gen_instance
        demands, distances = gen_instance(n_customers, device='cpu')
        instance = {"demands": demands, "distances": distances}
        test_instances.append(instance)

    # Save both train and test datasets to a file
    torch.save({'train_instances': train_instances, 'test_instances': test_instances}, file_path)
    print(f"Fixed CVRP train and test datasets saved to {file_path}")

# generate_and_save_fixed_cvrp_datasets(
#     n_customers=200, 
#     steps_per_epoch=128*2, 
#     max_epochs=10, 
#     test_size=30, 
#     file_path='cvrpdata200.pth'
# )


def generate_and_save_fixed_cvrp_test_dataset(n_customers, n_test_instances, k_sparse, file_path):

    test_instances = []
    
    for _ in range(n_test_instances):
        # Generate a test CVRP instance using gen_instance
        demands, distances = gen_instance(n_customers, device='cpu')  # Generate CVRP instance
        
        instance = {"demands": demands, "distances": distances}
        test_instances.append(instance)

    # Save the dataset to a file
    torch.save({'test_instances': test_instances}, file_path)
    print(f"Fixed CVRP test dataset saved to {file_path}")


# generate_and_save_fixed_cvrp_test_dataset(200,100,10,'cvrpdataTEST200.pth')