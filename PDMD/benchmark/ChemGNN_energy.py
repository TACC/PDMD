from torch.nn import Embedding, Linear
from torch_geometric.nn import BatchNorm, global_add_pool
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, ReLU, Sequential
from torch_geometric.nn.dense.linear import Linear
from PDMD.benchmark.utils import one_time_generate_forward_input_energy, reverse_min_max_scaler, molecular_energy, atomic_energy_map
from PDMD.benchmark.ChemGNN import CEALConv


class ChemGNN_EnergyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_emb = Embedding(20, 10)  # self.edge_emb = Embedding(4, 50)
        self.conv_num = 2
        self.in_num = 1262
        self.mid_num = 200
        self.out_num = 20
        aggregators = ['sum', 'mean', 'min', 'max', 'std']
        self.weights = torch.nn.Parameter(torch.rand(len(aggregators)))
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        for _ in range(self.conv_num):
            if _ == 0:
                conv = CEALConv(in_channels=self.in_num, out_channels=self.mid_num, weights=self.weights,
                                aggregators=aggregators, edge_dim=10, towers=1, pre_layers=3, post_layers=3,
                                divide_input=False)
                norms = BatchNorm(self.mid_num)
            else:
                conv = CEALConv(in_channels=self.mid_num, out_channels=self.out_num, weights=self.weights,
                                aggregators=aggregators, edge_dim=10, towers=1, pre_layers=3, post_layers=3,
                                divide_input=False)
                norms = BatchNorm(self.out_num)
            self.convs.append(conv)
            self.batch_norms.append(norms)

        self.pre_mlp = Sequential(Linear(1262, 1262), ReLU())
        self.energy_predictor = Sequential(Linear(self.out_num, 100), ReLU(), Linear(100, 10), ReLU(), Linear(10, 1))

    def forward(self, atomic_numbers, tensor_positions, energy_feature_min_values, energy_feature_max_values, neighborlist=None):

        # convert tensor positions to a numpy ndarray
        # positions = tensor_positions.detach().numpy()

        x, self.CMA = one_time_generate_forward_input_energy(atomic_numbers, tensor_positions, self.CMA, energy_feature_min_values, energy_feature_max_values, neighborlist)
        assert {"x", "edge_index", "edge_attr", "batch"}.issubset(x.keys())
        x, edge_index, edge_attr, batch = iter(
            [x.get(one_key) for one_key in ["x", "edge_index", "edge_attr", "batch"]])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_emb = self.edge_emb.to(device)
        self.batch_norms = self.batch_norms.to(device)
        self.convs = self.convs.to(device)
        self.pre_mlp = self.pre_mlp.to(device)
        self.energy_predictor = self.energy_predictor.to(device)

        edge_attr = self.edge_emb(edge_attr)
        # x = x.to(torch.float64)
        x = self.pre_mlp(x)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, self.weights, edge_attr)))

        x_energy = global_add_pool(x, batch)
        x_energy = self.energy_predictor(x_energy)

        energy = x_energy.cpu().detach().numpy()[0]
        energy = molecular_energy(atomic_numbers, energy, atomic_energy_map)
        #energy = reverse_min_max_scaler(energy)
        #energy = torch.tensor(energy)
        return energy



