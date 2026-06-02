from torch.nn import Embedding, Linear
from torch_geometric.nn import BatchNorm
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, ReLU, Sequential
from torch_geometric.nn.dense.linear import Linear
from PDMD.benchmark.utils import one_time_generate_forward_input_force
from PDMD.benchmark.HyperCEALConv import HyperCEALConv

class ChemGNN_ForcesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "ChemGNN_forces"
        self.conv_num = 2
        self.in_num = 722
        aggregators = ['sum', 'mean', 'min', 'max', 'std']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = torch.nn.Parameter(torch.rand(len(aggregators)))
        self.hyperedge_order = 2
        self.hyperedge_weights = torch.nn.Parameter(torch.rand(self.hyperedge_order - 1).to(device))
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        for _ in range(self.conv_num):
            if _ == 0:
                conv = HyperCEALConv(in_channels=self.in_num, out_channels=self.in_num, weights=self.weights,
                                aggregators=aggregators, edge_dim=10, towers=1, pre_layers=1, post_layers=1,
                                hyperedge_order=2, hyperedge_weights=self.hyperedge_weights,
                                divide_input=False)
                norms = BatchNorm(self.in_num)
            else:
                conv = HyperCEALConv(in_channels=self.in_num, out_channels=self.in_num, weights=self.weights,
                                aggregators=aggregators, edge_dim=10, towers=1, pre_layers=1, post_layers=1,
                                hyperedge_order=2, hyperedge_weights=self.hyperedge_weights,
                                divide_input=False)
                norms = BatchNorm(self.in_num)
            self.convs.append(conv.to(device))
            self.batch_norms.append(norms.to(device))
        self.node_embedding = Sequential(Linear(self.in_num, self.in_num), ReLU()).to(device)
        self.edge_embedding = Sequential(Linear(1, 32), ReLU(), Linear(32, 10)).to(device)
        self.force_predictor = Sequential(Linear(self.in_num, 300), ReLU(), Linear(300, 3)).to(device)

    def forward(self, atomic_numbers, positions, forces_feature_min_values, forces_feature_max_values, neighborlist_soap=None, neighborlist_chemgnn=None):
        x = one_time_generate_forward_input_force(atomic_numbers, positions, forces_feature_min_values, forces_feature_max_values, neighborlist_soap, neighborlist_chemgnn)
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

        force = self.force_predictor(node_attr).cpu()
        return force

