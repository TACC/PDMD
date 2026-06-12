from torch.nn import Embedding, Linear
from torch_geometric.nn import BatchNorm, global_add_pool
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, ReLU, Sequential
from torch_geometric.nn.dense.linear import Linear
from PDMD.benchmark.utils import one_time_generate_forward_input_energy, molecular_energy, atomic_energy_map
from PDMD.benchmark.HyperCEALConv import HyperCEALConv

class ChemGNN_EnergyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "ChemGNN_energy"
        self.conv_num = 2
        self.in_num = 1262
        self.mid_num = 200
        self.out_num = 20
        aggregators = ['sum', 'mean', 'min', 'max', 'std']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = torch.nn.Parameter(torch.rand(len(aggregators)).to(device))
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
            self.convs.append(conv.to(device))
            self.batch_norms.append(norms.to(device))
        self.node_embedding = Sequential(Linear(1262, 1262), ReLU()).to(device)
        self.edge_embedding = Sequential(Linear(1,10)).to(device)
        self.energy_predictor = Sequential(Linear(self.out_num, 100), ReLU(), Linear(100, 10), ReLU(), Linear(10, 1)).to(device)

    def forward(self, atomic_numbers, tensor_positions, energy_feature_min_values, energy_feature_max_values, neighborlist_soap=None, neighborlist_chemgnn=None):
        x = one_time_generate_forward_input_energy(atomic_numbers, tensor_positions, energy_feature_min_values, energy_feature_max_values, neighborlist_soap, neighborlist_chemgnn)
        assert {"x", "batch", "hyperedge_index", "hyperedge_attr"}.issubset(x.keys())
        node_attr, batch, hyperedge_index, hyperedge_attr = iter([x.get(one_key) for one_key in ["x", "batch", "hyperedge_index", "hyperedge_attr"]])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hyperedge_attr = hyperedge_attr.unsqueeze(-1).to(torch.get_default_dtype()).to(device)
        node_attr = node_attr.to(torch.get_default_dtype()).to(device)
        
        node_attr = self.node_embedding(node_attr)
        hyperedge_attr = self.edge_embedding(hyperedge_attr)
        weights = self.weights.to(device)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            node_attr = F.relu(batch_norm(conv(node_attr, hyperedge_index, weights, hyperedge_attr)))

        x_energy = global_add_pool(node_attr, batch)
        x_energy = self.energy_predictor(x_energy)

        energy = x_energy.cpu().detach().numpy()[0]
        energy = molecular_energy(atomic_numbers, energy, atomic_energy_map)

        return energy



