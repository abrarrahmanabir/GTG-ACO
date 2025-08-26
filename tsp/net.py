import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import torch_geometric.nn as gnn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
# GNN for edge embeddings
class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=2, units=32, act_fn='silu', agg_fn='mean'): # TODO feats=1
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(1, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.se_layer = SELayer(channel=units, reduction=16)

    def reset_parameters(self):
        raise NotImplementedError
    def forward(self, x, edge_index, edge_attr):
        x = x
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)
        for i in range(self.depth):
            x0 = x
            # x1 = self.v_lins1[i](x0)
            # x2 = self.v_lins2[i](x0)
            # x3 = self.v_lins3[i](x0)
            # x4 = self.v_lins4[i](x0)

            x1 = self.v_lins1[i](x0)
            
            x1 = x1.view(x1.size(0), x1.size(1), 1, 1)  # Reshape to (node, channels, 1, 1)
            x1 = self.se_layer(x1) 
            x1 = x1.view(x1.size(0), x1.size(1))  

            x2 = self.v_lins2[i](x0)

            x2 = x2.view(x2.size(0), x2.size(1), 1, 1)  # Reshape to (node, channels, 1, 1)
            x2 = self.se_layer(x2) 
            x2 = x2.view(x2.size(0), x2.size(1))  


            x3 = self.v_lins3[i](x0)
            x3 = x3.view(x3.size(0), x3.size(1), 1, 1)
            x3 = self.se_layer(x3)
            x3 = x3.view(x3.size(0), x3.size(1))

            x4 = self.v_lins4[i](x0)

            x4 = x4.view(x4.size(0), x4.size(1), 1, 1) 
            x4 = self.se_layer(x4)
            x4 = x4.view(x4.size(0), x4.size(1))





            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return w

# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device
    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad = False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                x = torch.sigmoid(x) # last layer
        return x

# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='silu'):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn)
    def forward(self, x):
        return super().forward(x).squeeze(dim = -1)
    

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_net = EmbNet()
        self.par_net_phe = ParNet()
        self.par_net_heu = ParNet()
    def forward(self, pyg):
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        #print("x shape is : ", x.shape)
        emb = self.emb_net(x, edge_index, edge_attr)
        heu = self.par_net_heu(emb)
        return heu
    
    def freeze_gnn(self):
        for param in self.emb_net.parameters():
            param.requires_grad = False
            
    @staticmethod
    def reshape(pyg, vector):
        '''Turn phe/heu vector into matrix with zero padding 
        '''
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix








def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize and count parameters
model = Net()
total_params = count_parameters(model)
print(total_params)