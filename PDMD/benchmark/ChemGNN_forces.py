from torch.nn import Embedding, Linear
from torch_geometric.nn import BatchNorm
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, ReLU, Sequential
from torch_geometric.nn.dense.linear import Linear
from PDMD.benchmark.utils import one_time_generate_forward_input_force
from PDMD.benchmark.ChemGNN import CEALConv


class ChemGNN_ForcesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_emb = Embedding(20, 10)
        self.conv_num = 2
        self.in_num = 722
        aggregators = ['sum', 'mean', 'min', 'max', 'std']
        self.weights = torch.nn.Parameter(torch.rand(len(aggregators)))
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        for _ in range(self.conv_num):
            if _ == 0:
                conv = CEALConv(in_channels=self.in_num, out_channels=self.in_num, weights=self.weights,
                                aggregators=aggregators, edge_dim=10, towers=1, pre_layers=1, post_layers=1,
                                divide_input=False)
                norms = BatchNorm(self.in_num)
            else:
                conv = CEALConv(in_channels=self.in_num, out_channels=self.in_num, weights=self.weights,
                                aggregators=aggregators, edge_dim=10, towers=1, pre_layers=1, post_layers=1,
                                divide_input=False)
                norms = BatchNorm(self.in_num)
            self.convs.append(conv)
            self.batch_norms.append(norms)

        self.pre_mlp = Sequential(Linear(self.in_num, self.in_num), ReLU())
        self.force_predictor = Sequential(Linear(self.in_num, 300), ReLU(), Linear(300, 3))

    def forward(self, atomic_numbers, positions, forces_feature_min_values, forces_feature_max_values):
        x, self.CMA = one_time_generate_forward_input_force(atomic_numbers, positions, self.CMA, forces_feature_min_values, forces_feature_max_values)
        assert {"x", "edge_index", "edge_attr", "batch"}.issubset(x.keys())
        x, edge_index, edge_attr, batch = iter([x.get(one_key) for one_key in ["x", "edge_index", "edge_attr", "batch"]])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_emb = self.edge_emb.to(device)
        self.batch_norms = self.batch_norms.to(device)
        self.convs = self.convs.to(device)
        self.pre_mlp = self.pre_mlp.to(device)
        self.force_predictor = self.force_predictor.to(device)

        agg_weights = self.weights
        edge_attr = self.edge_emb(edge_attr)
        # x = x.to(torch.float64)
        x = self.pre_mlp(x)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, agg_weights, edge_attr)))
        x_force = self.force_predictor(x)

        force = x_force.cpu()
        return force

