import pickle
import time
import torch
from tqdm import tqdm
from net import Net
from aco import ACO
from utils import *
from model import Net_tr

torch.manual_seed(1234)

lr = 3e-4
EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_fixed_datasets(file_path):
    with open(file_path, "rb") as f:
        dataset = pickle.load(f)
    return dataset['train_instances'], dataset['val_instances']

def train_instance(model, model2, optimizer1, optimizer2, pyg_data, due_time, weights, processing_time, n_ants):
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
        due_time=due_time,
        weights=weights,
        processing_time=processing_time,
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

def train_epoch(n_ants, epoch, steps_per_epoch, net, net2, optimizer1, optimizer2, train_instances):
    start_index = epoch * steps_per_epoch

    for step in range(steps_per_epoch):
        instance = train_instances[start_index + step]
        pyg_data = instance['pyg_data'].to(device)
        due_time = instance['due_time'].to(device)
        weights = instance['weights'].to(device)
        processing_time = instance['processing_time'].to(device)
        train_instance(net, net2, optimizer1, optimizer2, pyg_data, due_time, weights, processing_time, n_ants)

@torch.no_grad()
def validation(n_ants, epoch, net, net2, val_instances, animator=None):
    sum_bl, sum_sample_best = 0, 0
    n_val = len(val_instances)
    
    for instance in val_instances:
        pyg_data = instance['pyg_data'].to(device)
        due_time = instance['due_time'].to(device)
        weights = instance['weights'].to(device)
        processing_time = instance['processing_time'].to(device)

        bl, sample_best = infer_instance(len(due_time), net, net2, n_ants, pyg_data, due_time, weights, processing_time)
        sum_bl += bl
        sum_sample_best += sample_best
        
    avg_bl = sum_bl / n_val
    avg_sample_best = sum_sample_best / n_val
    
    if animator:
        animator.add(epoch + 1, (avg_bl, avg_sample_best))
        
    return avg_bl, avg_sample_best

@torch.no_grad()
def infer_instance(n, model, model2, n_ants, pyg_data, due_time, weights, processing_time):
    model.eval()
    model2.eval()

    heu_vec = model(pyg_data)  # Forward pass through the heuristic model
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS

    pheromone_update_vec = model2(pyg_data)  # Forward pass through the pheromone model
    pheromone_update_mat = model2.reshape(pyg_data, pheromone_update_vec) + EPS

    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat,
        pheromone_guide=pheromone_update_mat,
        due_time=due_time,
        weights=weights,
        processing_time=processing_time,
        device=device
    )
    
    costs, log_probs = aco.sample()
    aco.run(n_iterations=T)
    baseline = costs.mean()
    best_sample_cost = torch.min(costs)
    
    return baseline.item(), best_sample_cost.item()

def train(mode, n_ants, steps_per_epoch, epochs, dataset_path):
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

    avg_bl, avg_best = validation(n_ants, -1, net, net2, val_instances, animator)

    val_results = [(avg_bl, avg_best)]

    sum_time = 0
    for epoch in tqdm(range(epochs)):
        start = time.time()
        train_epoch(n_ants, epoch, steps_per_epoch, net, net2, optimizer1, optimizer2, train_instances)
        sum_time += time.time() - start
        
        avg_bl, avg_sample_best = validation(n_ants, epoch, net, net2, val_instances, animator)
        val_results.append((avg_bl, avg_sample_best))

    
    for epoch in range(-1, epochs):
        print(f'Epoch {epoch}:', val_results[epoch + 1])
    
    return net, net2


n_node, n_ants = 100 , 20
k_sparse = 10
steps_per_epoch = 256
T = 50
epochs = 20
dataset_path = 'smtwtp/train_val_datasets-100.pkl'  

gnn, phero = train("tr", n_ants, steps_per_epoch, epochs, dataset_path)

def infer_instance(model,model2, instance, n_ants, t_aco_diff):

    pyg_data, due_time, weights, processing_time = instance

    if True:

        model.eval()
        model2.eval()

        heu_vec = model(pyg_data)  
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS

        pheromone_update_vec = model2(pyg_data) 
        pheromone_update_mat = model2.reshape(pyg_data, pheromone_update_vec) + EPS

        aco = ACO(
            n_ants=n_ants,
            heuristic=heu_mat,
            pheromone_guide=pheromone_update_mat,
            due_time=due_time,
            weights=weights,
            processing_time=processing_time,
            device=device
        )


    results = torch.zeros(size=(len(t_aco_diff),), device=device)
    for i, t in enumerate(t_aco_diff):
        best_cost = aco.run(t)
        results[i] = best_cost
    return results

@torch.no_grad()
def test(dataset, model,model2, n_ants, t_aco):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i+1]-_t_aco[i] for i in range(len(_t_aco)-1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),), device=device)
    start = time.time()
    for instance in dataset:
        results = infer_instance(model,model2, instance, n_ants, t_aco_diff)
        sum_results += results
    end = time.time()
    
    return sum_results / len(dataset), end-start

t_aco = [1,10,20 , 30, 40 ,50, 100]
test_list = load_test_dataset(n_node, device)

avg_aco_best, duration = test(test_list, gnn.to(device),phero.to(device) ,n_ants, t_aco)
for i, t in enumerate(t_aco):
    print(f"T={t}, average cost is {avg_aco_best[i]}.")
