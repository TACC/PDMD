import torch.nn.functional as F
from torch.nn import Embedding, Linear
from torch_geometric.nn import BatchNorm, global_add_pool
from typing import Dict, Callable, List, Optional, Set, get_type_hints
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import ModuleList, ReLU, Sequential, BatchNorm1d
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.nn.inits import reset
from PDMD.models.HyperCEALConv import HyperCEALConv

class ENERGY_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "ChemGNN_energy"
        self.conv_num = 2
        self.in_num = 1262
        self.mid_num = 200
        self.out_num = 20
        aggregators = ['sum', 'mean', 'min', 'max', 'std']
        self.weights = torch.nn.Parameter(torch.rand(len(aggregators)))
        self.hyperedge_order = 3
        self.hyperedge_weights = torch.nn.Parameter(torch.rand(self.hyperedge_order - 1))
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        for _ in range(self.conv_num):
            if _ == 0:
                conv = HyperCEALConv(in_channels=self.in_num, out_channels=self.mid_num, weights=self.weights,
                                aggregators=aggregators, edge_dim=10, towers=1, pre_layers=3, post_layers=3,
                                hyperedge_order=self.hyperedge_order, hyperedge_weights=self.hyperedge_weights,
                                divide_input=False)
                norms = BatchNorm(self.mid_num)
            else:
                conv = HyperCEALConv(in_channels=self.mid_num, out_channels=self.out_num, weights=self.weights,
                                aggregators=aggregators, edge_dim=10, towers=1, pre_layers=3, post_layers=3,
                                hyperedge_order=self.hyperedge_order, hyperedge_weights=self.hyperedge_weights,
                                divide_input=False)
                norms = BatchNorm(self.out_num)
            self.convs.append(conv)
            self.batch_norms.append(norms)
        self.node_embedding = Sequential(Linear(1262, 1262), ReLU())
        self.edge_embedding = Sequential(Linear(1,10))
        self.energy_predictor = Sequential(Linear(self.out_num, 100), ReLU(), Linear(100, 10), ReLU(), Linear(10, 1))

    def forward(self, input_dict):
        assert {"x", "batch", "hyperedge_index", "hyperedge_attr"}.issubset(input_dict.keys())
        node_attr, batch, hyperedge_index, hyperedge_attr = iter([input_dict.get(one_key) for one_key in ["x", "batch", "hyperedge_index", "hyperedge_attr"]])

        hyperedge_attr = hyperedge_attr.unsqueeze(-1)
        if torch.get_default_dtype() == torch.float64:
            node_attr = node_attr.to(torch.float64)
            hyperedge_attr = hyperedge_attr.to(torch.float64)
        node_attr = self.node_embedding(node_attr)
        hyperedge_attr = self.edge_embedding(hyperedge_attr)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            node_attr = F.relu(batch_norm(conv(node_attr, hyperedge_index, self.weights, hyperedge_attr)))

        energy = global_add_pool(node_attr, batch)
        energy = self.energy_predictor(energy)
        return energy
