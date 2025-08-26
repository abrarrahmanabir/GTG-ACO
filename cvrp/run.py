import time
import torch
from torch.distributions import Categorical, kl
from tqdm import tqdm
from net import Net
from model import Net_tr
from aco import ACO
from utils import gen_pyg_data
torch.manual_seed(1234)

lr = 3e-4
EPS = 1e-10

device = 'cuda:0'
torch.autograd.set_detect_anomaly(True)


def load_fixed_datasets(file_path):
    dataset = torch.load(file_path)
    return dataset['train_instances'], dataset['test_instances']

def train_instance(model, model2, optimizer1, optimizer2, pyg_data, distances, demands, n_ants):
    model.train()
    model2.train()
    
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS

    pheromone_update_vec = model2(pyg_data)
    pheromone_update_mat = model2.reshape(pyg_data, pheromone_update_vec) + EPS

    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat,
        pheromone_guide=pheromone_update_mat,  
        distances=distances,
        demand=demands,
        device=device
    )
    
    costs, log_probs = aco.sample()
    baseline = costs.mean()
    joint_loss = torch.sum((costs - baseline) * log_probs.sum(dim=0)) / aco.n_ants
    
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    joint_loss.backward()
    optimizer1.step()
    optimizer2.step()

def train_epoch(n_node, n_ants, epoch, steps_per_epoch, net, net2, optimizer1, optimizer2, flag, instances):
    start_index = epoch * steps_per_epoch

    for step in range(steps_per_epoch):
        instance = instances[start_index + step]
        
        demands = instance['demands'].to(device)
        distances = instance['distances'].to(device)
        
        pyg_data = gen_pyg_data(demands, distances, device)
        
        train_instance(net, net2, optimizer1, optimizer2, pyg_data, distances, demands, n_ants)

@torch.no_grad()
def validation(n_ants, epoch, net, net2, test_instances, animator=None):
    sum_bl, sum_sample_best, sum_aco_best = 0, 0, 0
    n_val = len(test_instances)
    
    for instance in test_instances:
        demands = instance['demands'].to(device)
        distances = instance['distances'].to(device)
        
        pyg_data = gen_pyg_data(demands, distances, device)
        
        bl, sample_best, aco_best = infer_instance(len(demands), net, net2, n_ants, demands, distances)
        sum_bl += bl
        sum_sample_best += sample_best
        sum_aco_best += aco_best
        
    avg_bl = sum_bl / n_val
    avg_sample_best = sum_sample_best / n_val
    avg_aco_best = sum_aco_best / n_val
    
    if animator:
        animator.add(epoch + 1, (avg_bl, avg_sample_best, avg_aco_best))
        
    return avg_bl, avg_sample_best, avg_aco_best

@torch.no_grad()
def infer_instance(n, model, model2, n_ants, demands, distances):
    model.eval()
    model2.eval()
    
    pyg_data = gen_pyg_data(demands, distances, device)
    
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS

    pheromone_update_vec = model2(pyg_data)
    pheromone_update_mat = model2.reshape(pyg_data, pheromone_update_vec) + EPS

    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat,
        pheromone_guide=pheromone_update_mat,
        distances=distances,
        demand=demands,
        device=device
    )
    
    costs, log_probs = aco.sample()
    aco.run(n_iterations=T)
    baseline = costs.mean()
    best_sample_cost = torch.min(costs)
    best_aco_cost = aco.lowest_cost
    
    return baseline.item(), best_sample_cost.item(), best_aco_cost.item()

def train(mode, n_node, k_sparse, n_ants, steps_per_epoch, epochs, flag, dataset_path):
    
    if mode == "gnn":
        net = Net().to(device)
        net2 = Net().to(device)
    elif mode == "tr":
        net = Net_tr().to(device)
        net2 = Net_tr().to(device)

    optimizer1 = torch.optim.AdamW(net.parameters(), lr=lr)
    optimizer2 = torch.optim.AdamW(net2.parameters(), lr=lr)

    
    train_instances, val_instances = load_fixed_datasets(dataset_path)

    
    animator = None
    avg_bl, avg_best, avg_aco_best = validation(n_ants, -1, net, net2, val_instances, animator)
    val_results = [(avg_bl, avg_best, avg_aco_best)]
    
    sum_time = 0
    for epoch in tqdm(range(epochs)):
        start = time.time()
        train_epoch(n_node, n_ants, epoch, steps_per_epoch, net, net2, optimizer1, optimizer2, flag, train_instances)
        sum_time += time.time() - start
        
        avg_bl, avg_sample_best, avg_aco_best = validation(n_ants, epoch, net, net2, val_instances, animator)
        val_results.append((avg_bl, avg_sample_best, avg_aco_best))
            
    for epoch in range(-1, epochs):
        print(f'Epoch {epoch}:', val_results[epoch + 1])
    
    return net, net2



epochs , steps_per_epoch = 1 , 2
n_ants= 20
T = 30


gnn , phero = train(
    mode="tr", 
    n_node=200, 
    k_sparse=10, 
    n_ants=n_ants, 
    steps_per_epoch=steps_per_epoch, 
    epochs=epochs, 
    flag=True, 
    dataset_path='./cvrp/cvrpdata100.pth'
)

def load_fixed_test_dataset(file_path):
    dataset = torch.load(file_path)
    return dataset['test_instances']


@torch.no_grad()
def infer_instance(model, model2, demands, distances, n_ants, t_aco_diff):
    model.eval()
    model2.eval()
    
    pyg_data = gen_pyg_data(demands, distances, device)
    
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS

    pheromone_update_vec = model2(pyg_data)
    pheromone_update_mat = model2.reshape(pyg_data, pheromone_update_vec) + EPS

    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat,
        pheromone_guide=pheromone_update_mat,
        distances=distances,
        demand=demands,
        device=device
    )
    
    aco_results = []

    for t in t_aco_diff:
        aco.run(n_iterations=t)
        aco_results.append(aco.lowest_cost)

    return torch.tensor(aco_results, device=device)

@torch.no_grad()
def test(dataset, model, model2, n_ants, t_aco):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i+1] - _t_aco[i] for i in range(len(_t_aco) - 1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),), device=device)
    
    start = time.time()
    for instance in dataset:
        demands = instance['demands'].to(device)
        distances = instance['distances'].to(device)
        results = infer_instance(model, model2, demands, distances, n_ants, t_aco_diff)
        sum_results += results
    end = time.time()
    
    return sum_results / len(dataset), end - start

test_list = load_fixed_test_dataset('.\cvrp\cvrpdataTEST200.pth')
t_aco = [ 10, 20, 30, 40, 50, 100]
avg_aco_best, duration = test(test_list, gnn.to(device), phero.to(device), n_ants, t_aco)
for i, t in enumerate(t_aco):
    print(f"T={t}, average cost is {avg_aco_best[i]}.")











