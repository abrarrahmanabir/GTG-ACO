import time
import torch
from torch.distributions import Categorical, kl
from tqdm import tqdm
from bpp.model import Net_tr
from net import Net
from aco import ACO
from utils import gen_pyg_data, gen_instance, load_test_dataset

torch.manual_seed(1234)

lr = 3e-4
EPS = 1e-10
device = 'cuda'

def train_instance(n, model, model2, optimizer1, optimizer2, n_ants):
    model.train()
    model2.train()
    
    demands = gen_instance(n, device).to(device)  # Ensure demands are on the correct device
    pyg_data = gen_pyg_data(demands, device).to(device)  # Ensure PyG data is on the correct device

    # Forward pass through the heuristic model
    heu_vec = model(pyg_data)
    heu_mat = heu_vec.reshape((n+1, n+1)) + EPS

    # Forward pass through the pheromone model
    pheromone_update_vec = model2(pyg_data)
    pheromone_update_mat = pheromone_update_vec.reshape((n+1, n+1)) + EPS

    # Create the ACO instance using both models
    aco = ACO(
        demand=demands,
        n_ants=n_ants,
        heuristic=heu_mat,
        pheromone_guide=pheromone_update_mat,
        device=device
    )

    # Sample paths and calculate costs
    costs, log_probs = aco.sample()
    baseline = costs.mean()
    joint_loss = torch.sum((costs.to(device) - baseline.to(device)) * log_probs.to(device).sum(dim=0)) / aco.n_ants

    
    # Optimize both models
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    joint_loss.backward()
    optimizer1.step()
    optimizer2.step()

def infer_instance(n, model, model2, n_ants):
    model.eval()
    model2.eval()

    demands = gen_instance(n, device).to(device)  # Ensure demands are on the correct device
    pyg_data = gen_pyg_data(demands, device).to(device)  # Ensure PyG data is on the correct device

    # Forward pass through the heuristic model
    heu_vec = model(pyg_data)
    heu_mat = heu_vec.reshape((n+1, n+1)) + EPS

    # Forward pass through the pheromone model
    pheromone_update_vec = model2(pyg_data)
    pheromone_update_mat = pheromone_update_vec.reshape((n+1, n+1)) + EPS

    # Create the ACO instance using both models
    aco = ACO(
        demand=demands,
        n_ants=n_ants,
        heuristic=heu_mat,
        pheromone_guide=pheromone_update_mat,
        device=device
    )

    # Sample paths and run ACO
    costs, log_probs = aco.sample()
    aco.run(n_iterations=T)
    
    # Calculate results
    baseline = costs.mean()
    best_sample_cost = torch.min(costs)
    best_aco_cost = -aco.best_fitness  
    
    return baseline.item(), best_sample_cost.item(), best_aco_cost.item()


def train_epoch(n_node, n_ants, steps_per_epoch, net, net2, optimizer1, optimizer2):
    for _ in range(steps_per_epoch):
        train_instance(n_node, net, net2, optimizer1, optimizer2, n_ants)

@torch.no_grad()
def validation(n_node, n_ants, epoch, net, net2, animator=None):
    sum_bl, sum_sample_best, sum_aco_best = 0, 0, 0
    n_val = 100
    for _ in range(n_val):
        bl, sample_best, aco_best = infer_instance(n_node, net, net2, n_ants)
        sum_bl += bl
        sum_sample_best += sample_best
        sum_aco_best += aco_best
    
    # Calculate averages
    avg_bl = sum_bl / n_val
    avg_sample_best = sum_sample_best / n_val
    avg_aco_best = sum_aco_best / n_val
    
    if animator:
        animator.add(epoch + 1, (avg_bl, avg_sample_best, avg_aco_best))
    
    return avg_bl, avg_sample_best, avg_aco_best

def train(mode, n_node, n_ants, steps_per_epoch, epochs):
    if mode == "gnn":
        net = Net().to(device)
        net2 = Net().to(device)
    elif mode == "tr":
        net = Net_tr().to(device)
        net2 = Net_tr().to(device)

    # Optimizers for both models
    optimizer1 = torch.optim.AdamW(net.parameters(), lr=lr)
    optimizer2 = torch.optim.AdamW(net2.parameters(), lr=lr)

    # Initial validation before training
    animator = None
    avg_bl, avg_best, avg_aco_best = validation(n_node, n_ants, -1, net, net2, animator)
    val_results = [(avg_bl, avg_best, avg_aco_best)]

    # Training loop
    sum_time = 0
    for epoch in tqdm(range(epochs)):
        start = time.time()
        train_epoch(n_node, n_ants, steps_per_epoch, net, net2, optimizer1, optimizer2)
        sum_time += time.time() - start
        
        # Validation after each epoch
        avg_bl, avg_best, avg_aco_best = validation(n_node, n_ants, epoch, net, net2, animator)
        val_results.append((avg_bl, avg_best, avg_aco_best))
    
    for epoch in range(-1, epochs):
        print(f'Epoch {epoch}:', val_results[epoch + 1])

    return net, net2



n_node, n_ants = 120, 20
steps_per_epoch = 256
epochs = 10
T = 50
gnn,phero = train("tr", n_node, n_ants, steps_per_epoch, epochs)






def infer_instance(n, model, model2, demands, n_ants, t_aco_diff):

    
    # Ensure that demands are on the correct device
    demands = demands.to(device)
  

    # Generate pyg_data and ensure it is on the correct device
    pyg_data = gen_pyg_data(demands, device)

    # Forward pass through the heuristic model (move inputs to the correct device)
    heu_vec = model(pyg_data)
   
    
    heu_mat = heu_vec.reshape((n+1, n+1)) + EPS
   

    # Forward pass through the pheromone model (move inputs to the correct device)
    pheromone_update_vec = model2(pyg_data)
 

    pheromone_update_mat = pheromone_update_vec.reshape((n+1, n+1)) + EPS
   

    # Create the ACO instance using both models
    aco = ACO(
        demand=demands,
        n_ants=n_ants,
        heuristic=heu_mat,
        pheromone_guide=pheromone_update_mat,
        device=device  # Ensure ACO works on the same device
    )

    # Initialize results tensor on the correct device
    results = torch.zeros(size=(len(t_aco_diff),), device=device)
    
    # Run ACO for different iterations and record the best cost
    for i, t in enumerate(t_aco_diff):
        best_cost = aco.run(t)
        results[i] = best_cost
    
    
    return results


@torch.no_grad()
def test(n_node, dataset, model, model2, n_ants, t_aco):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i+1] - _t_aco[i] for i in range(len(_t_aco) - 1)]

    # Initialize sum_results tensor on the same device
    sum_results = torch.zeros(size=(len(t_aco_diff),), device=device)

    # Measure the total time for testing
    start = time.time()
    
    for demands in dataset:
        # Ensure demands are on the correct device
        demands = demands.to(device)
        results = infer_instance(n_node, model, model2, demands, n_ants, t_aco_diff)
        sum_results += results
    
    end = time.time()
    
    return sum_results / len(dataset), end - start

# Running the test
n_ants = 20
t_aco = [1, 10, 20, 30, 40, 50, 100]
n_node = 120
# T = 5
test_list = load_test_dataset(n_node, device)
avg_aco_best, duration = test(n_node, test_list[:10], gnn.to(device), phero.to(device), n_ants, t_aco)
for i, t in enumerate(t_aco):
    print(f"T={t}, average objective is {avg_aco_best[i]}.")
