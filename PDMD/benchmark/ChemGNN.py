from torch.nn import Linear
from typing import List, Optional
import torch
from torch import Tensor
from torch.nn import ModuleList, ReLU, Sequential
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.inits import reset


class CEALConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, weights: Tensor,
                 aggregators: List[str],edge_dim: Optional[int] = None, towers: int = 1,
                 pre_layers: int = 1, post_layers: int = 1,
                 divide_input: bool = False, **kwargs):

        kwargs.setdefault('aggr', None)
        super().__init__(node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        device = torch.device("cuda" if torch.cuda.is_available() and 'cuda' else "cpu")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input
        self.weights = weights

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers


        if self.edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, weights: Tensor,
                edge_attr: OptTensor = None
                ) -> Tensor:
        """"""
        device = torch.device("cuda" if torch.cuda.is_available() and 'cuda' else "cpu")
        self.pre_nns = self.pre_nns.to(device)
        self.post_nns = self.post_nns.to(device)
        self.edge_encoder = self.edge_encoder.to(device)
        self.lin = self.lin.to(device)

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, weights=weights, size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def aggregate(self, inputs: Tensor, index: Tensor, weights: Tensor,
                  dim_size: Optional[int] = None,
                  ) -> Tensor:

        device = torch.device("cuda" if torch.cuda.is_available() and 'cuda' else "cpu")
        outs = []

        #range in the index tensor
        max_index = torch.max(index).item()
        min_index = torch.min(index).item()
        unique_index = max_index - min_index + 1
        #the second and third dimension sizes of the inputs tensor
        ydim_inputs = inputs.size(dim=1)
        zdim_inputs = inputs.size(dim=2)
        #expand the one-dimensional index tensor to the same size as the three-dimensional inputs tensor
        #note that pytorch_scatter_reduce requires the same dimension for the index and inputs tesors
        tensor_index = index.unsqueeze(-1).unsqueeze(-1).expand(inputs.size())

        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = torch.zeros(unique_index, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
                out = weights[0] * out.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="sum", include_self=False)
                # out = weights[0] * scatter(inputs, index, 0, None, dim_size, reduce='sum')
                # out = weights[0] * scatter_sum(inputs, index, 0, None, dim_size=dim_size)
            elif aggregator == 'mean':
                out = torch.zeros(unique_index, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
                out = weights[1] * out.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="mean", include_self=False)
                # out = weights[1] * scatter(inputs, index, 0, None, dim_size, reduce='mean')
                # out = weights[1] * scatter_mean(inputs, index, 0, None, dim_size=dim_size)
            elif aggregator == 'min':
                out = torch.zeros(unique_index, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
                out = weights[2] * out.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="amin", include_self=False)
                # out = weights[2] * scatter(inputs, index, 0, None, dim_size, reduce='min')
                # out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = torch.zeros(unique_index, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
                out = weights[3] * out.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="amax", include_self=False)
                # out = weights[3] * scatter(inputs, index, 0, None, dim_size, reduce='max')
                # out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean =  torch.zeros(unique_index, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
                mean = mean.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="mean", include_self=False)
                mean_squares = torch.zeros(unique_index, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
                mean_squares = mean_squares.scatter_reduce(dim=0, index=tensor_index, src=inputs * inputs, reduce="mean", include_self=False)
                # mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                # mean_squares = scatter(inputs * inputs, index, 0, None,
                #                       dim_size, reduce='mean')
                out = (mean_squares - mean * mean)
                if aggregator == 'std':
                    out = weights[4] * torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)
        out = torch.cat(outs, dim=-1)

        outs = []
        outs.append(out)
        return torch.cat(outs, dim=-1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'edge_dim={self.edge_dim})')
