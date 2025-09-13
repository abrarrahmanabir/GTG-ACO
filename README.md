# GTG-ACO : Graph Transformer Guided Ant Colony Optimization for Learning Heuristics and Pheromone Dynamics for Combinatorial Optimization

This repository contains the implementation of GTG-ACO, a deep learning framework combining Graph Transformer models with Ant Colony Optimization (ACO) for solving combinatorial optimization problems.

### Dependencies

- Python 3.11.9
- CUDA 12.9
- PyTorch 2.5.1
- [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) 2.0.7
- [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) 0.6.9
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) 2.6.1
- d2l
- [networkx](https://networkx.org/) 2.8.4
- [numpy](https://numpy.org/) 1.23.3
- [numba](https://numba.pydata.org/) 0.56.4

## Installation
```bash
git clone https://github.com/abrarrahmanabir/GTG-ACO.git
cd GTG-ACO
```

### 📌 Available Combinatorial Optimization Problems

- **Traveling Salesman Problem (TSP):** see [`tsp/`]
- **Capacitated Vehicle Routing Problem (CVRP):** see [`cvrp/`]
- **Single Machine Total Weighted Tardiness Problem (SMTWTP):** see [`smtwtp/`]
- **Bin Packing Problem (BPP):** see [`bpp/`]


## 📂 Repository Structure

Every problem folder (e.g., `bpp/`, `tsp/`, `cvrp/`, `smtwtp/`) is organized in the same way:

\│── aco.py     # Core Ant Colony Optimization logic  
\│── model.py   # GTG-ACO model architecture  
\│── net.py     
\│── run.py     # Main script for model training  
\│── utils.py   # Utilities for generating random problem instances  

---

##  Training

To train the model on a selected problem instance, simply run:

python run.py



## 🔗 Acknowledgement

This codebase builds upon the DeepACO framework. We gratefully acknowledge their contributions and recommend citing their work if you use this repository:

> **DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization**. NeurIPS 2023 [https://proceedings.neurips.cc/paper_files/paper/2023/hash/883105b282fe15275991b411e6b200c5-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2023/hash/883105b282fe15275991b411e6b200c5-Abstract-Conference.html), 2023.  

