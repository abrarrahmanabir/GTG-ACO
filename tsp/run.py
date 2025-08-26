import time
import torch
from torch.distributions import Categorical, kl
from net import Net
from model import Net_tr
from aco import ACO
from utils import gen_pyg_data, load_val_dataset,load_test_dataset

torch.manual_seed(1234)

lr = 3e-4
EPS = 1e-10
device = 'cuda:0'
torch.autograd.set_detect_anomaly(True)


def load_fixed_dataset(file_path):
    dataset = torch.load(file_path)
    instances = dataset['instances']
    return instances



def train_instance(model, model2, optimizer1, optimizer2, pyg_data, distances, n_ants):
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





def infer_instance(model, pyg_data, distances, n_ants):
    model.eval()
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat,
        distances=distances,
        device=device
        )
    costs, log_probs = aco.sample()
    aco.run(n_iterations=T)
    baseline = costs.mean()
    best_sample_cost = torch.min(costs)
    best_aco_cost = aco.lowest_cost
    return baseline.item(), best_sample_cost.item(), best_aco_cost.item()




def train_epoch(n_node, n_ants, k_sparse, epoch, steps_per_epoch, net, net2, optimizer, optimizer2, flag, instances):

    start_index = epoch * steps_per_epoch

    for step in range(steps_per_epoch):

        instance = instances[start_index + step]
        
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse)
    
        train_instance(net, net2, optimizer, optimizer2, data, distances, n_ants)



def train(mode, n_node, k_sparse, n_ants, steps_per_epoch, epochs, flag, dataset_path):
    
    if mode == "gnn":
        print("GNN ")
        net = Net().to(device)
        net2 = Net().to(device)

    if mode == "tr":
        net = Net_tr().to(device)
        net2 = Net().to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    optimizer2 = torch.optim.AdamW(net2.parameters(), lr=lr)
    
    instances = load_fixed_dataset(dataset_path)
    
    val_list = load_val_dataset(n_node, k_sparse, device)
    
    sum_time = 0
    for epoch in range(epochs):
        start = time.time()
        train_epoch(n_node, n_ants, k_sparse, epoch, steps_per_epoch, net, net2, optimizer, optimizer2, flag, instances)
        sum_time += time.time() - start

    return net, net2


n_node, n_ants = 100 , 20
k_sparse = 10
steps_per_epoch = 128
epochs = 5
T=5
gnn , phero = train("gnn", n_node , k_sparse, n_ants, steps_per_epoch, epochs , flag=True , dataset_path='tsp/traindata100')



def infer_instance_test(model, model2, pyg_data, distances, n_ants, fine_tune=True, fine_tune_steps=64):

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
        distances=distances,
        device=device
        )
    costs, log_probs = aco.sample()
    aco.run(n_iterations=T)
    best_aco_cost = aco.lowest_cost
    return  best_aco_cost.item()

def test(dataset, model,model2, n_ants, t_aco, finetune):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i+1]-_t_aco[i] for i in range(len(_t_aco)-1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),), device=device)
    start = time.time()
    for pyg_data, distances in dataset:
        results = infer_instance_test(model,model2, pyg_data, distances, n_ants, finetune)
        sum_results += results
    end = time.time()
    
    return sum_results / len(dataset), end-start




k_sparse = 10
t_aco = [1, 10, 20, 30, 40, 50, 100]
test_list = load_test_dataset(n_node, k_sparse, device)
avg_aco_best, duration = test(test_list, gnn.to(device),  phero.to(device) , n_ants, t_aco, False)
for i, t in enumerate(t_aco):
    print("T={}, average cost is {}.".format(t, avg_aco_best[i]))    





