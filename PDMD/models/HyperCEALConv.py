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

class HyperCEALConv(MessagePassing):
    r"""Chemical Environment Adaptive Learning
    from the `"Chemical Environment Adaptive Learning for Optical Band Gap Prediction of Doped Graphitic Carbon Nitride Nanosheets"

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        aggregators (list of str): Set of aggregation function identifiers,
            namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"var"` and :obj:`"std"`.
        weights (Tensor): Weights for aggregators. 
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        towers (int, optional): Number of towers (default: :obj:`1`).
        pre_layers (int, optional): Number of transformation layers before
            aggregation (default: :obj:`1`).
        post_layers (int, optional): Number of transformation layers after
            aggregation (default: :obj:`1`).
        hyperedge_order (int, optional): Highest order of hyperedge 
            (default: :obj:`2`).
        divide_input (bool, optional): Whether the input features should
            be split between towers or not (default: :obj:`False`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    """

    def __init__(self, in_channels: int, out_channels: int, weights: Tensor,
                 aggregators: List[str],edge_dim: Optional[int] = None, towers: int = 1,
                 pre_layers: int = 1, post_layers: int = 1,
                 hyperedge_order: int = 2, hyperedge_weights: Tensor = None, 
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
        self.hyperedge_order = hyperedge_order 
        self.hyperedge_weights = hyperedge_weights
        self.divide_input = divide_input
        self.weights = weights

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        if self.edge_dim is not None:
            edge_encoder = []
            for _ in range(2, hyperedge_order + 1):
                edge_encoder += [Linear(edge_dim, self.F_in)]
            self.edge_encoder = ModuleList(edge_encoder)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules)).to(device)

            in_channels = (len(aggregators) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules)).to(device)

        self.lin = Linear(out_channels, out_channels).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            for enc in self.edge_encoder:
                enc.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Adj, weights: Tensor,
                hyperedge_attr: OptTensor = None,
                num_edges: Optional[int] = None
                ) -> Tensor:
        """"""

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        pairs, hyperedge_ids = self.hyperedge_to_pairs(hyperedge_index)
        out = self.propagate(pairs, 
                             x=x, 
                             hyperedge_index=hyperedge_index,
                             hyperedge_attr=hyperedge_attr, 
                             hyperedge_ids=hyperedge_ids, 
                             weights=weights, 
                             size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_index: Tensor,
                hyperedge_index: OptTensor, hyperedge_attr: OptTensor,
                hyperedge_ids: OptTensor) -> Tensor:

        h: Tensor = x_i
        if hyperedge_attr is not None:
            hyperedge_attr = self._edge_encode(hyperedge_attr, hyperedge_index)
            hyperedge_attr = hyperedge_attr.view(-1, 1, self.F_in)
            hyperedge_attr = hyperedge_attr.repeat(1, self.towers, 1)

            # pair attributes = sum of attributes of all hyperedges connecting the pair nodes 
            num_pairs = edge_index.size(1)
            edge_dim = hyperedge_attr.shape[1:]
            mask = hyperedge_ids >= 0
            he_ids = torch.arange(num_pairs, device=hyperedge_ids.device)
            he_ids = he_ids.unsqueeze(1).expand_as(hyperedge_ids)[mask]
            node_ids = hyperedge_ids[mask]
            pair_attr = torch.zeros(num_pairs, *edge_dim,
                     device=hyperedge_attr.device, dtype=hyperedge_attr.dtype)
            pair_attr.index_add_(0, he_ids, hyperedge_attr[node_ids])
            hyperedge_attr = pair_attr
            
            h = torch.cat([x_i, x_j, hyperedge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def aggregate(self, inputs: Tensor, index: Tensor, weights: Tensor,
                  dim_size: Optional[int] = None,
                  ) -> Tensor:

        device = torch.device("cuda" if torch.cuda.is_available() and 'cuda' else "cpu")
        outs = []

        #the second and third dimension sizes of the inputs tensor
        ydim_inputs = inputs.size(dim=1)
        zdim_inputs = inputs.size(dim=2)
        #expand the one-dimensional index tensor to the same size as the three-dimensional inputs tensor
        #note that pytorch_scatter_reduce requires the same dimension for the index and inputs tesors
        tensor_index = index.unsqueeze(-1).unsqueeze(-1).expand(inputs.size())

        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = torch.zeros(dim_size, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
                out = weights[0] * out.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="sum", include_self=False)
            elif aggregator == 'mean':
                out = torch.zeros(dim_size, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
                out = weights[1] * out.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="mean", include_self=False)
            elif aggregator == 'min':
                out = torch.zeros(dim_size, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
                out = weights[2] * out.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="amin", include_self=False)
            elif aggregator == 'max':
                out = torch.zeros(dim_size, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
                out = weights[3] * out.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="amax", include_self=False)
            elif aggregator == 'var' or aggregator == 'std':
                mean =  torch.zeros(dim_size, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
                mean = mean.scatter_reduce(dim=0, index=tensor_index, src=inputs, reduce="mean", include_self=False)
                mean_squares = torch.zeros(dim_size, ydim_inputs, zdim_inputs, dtype=inputs.dtype).to(device)
                mean_squares = mean_squares.scatter_reduce(dim=0, index=tensor_index, src=inputs * inputs, reduce="mean", include_self=False)
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

    def _edge_encode(self, hyperedge_attr: Tensor,
                     hyperedge_index: Tensor) -> Tensor:
        """Apply order-specific encoders to hyperedge attributes."""
        hedge_idx = hyperedge_index[1]
        n_hedge_attr = hyperedge_attr.size(0)
        n_horder = torch.bincount(hedge_idx,
                                  minlength=n_hedge_attr).to(torch.int32) - 2
        ha = torch.zeros(n_hedge_attr, self.F_in,
                         device=hyperedge_attr.device,
                         dtype=hyperedge_attr.dtype)
        for i in range(self.hyperedge_order - 1):
            mask = (n_horder == i)
            if mask.any():
                ha[mask] = self.hyperedge_weights[i]*self.edge_encoder[i](hyperedge_attr[mask])
        return ha

    def hyperedge_to_pairs(self, H: torch.Tensor):
        assert H.dim() == 2 and H.shape[0] == 2, "H must be a 2xN tensor"
        device = H.device
        nodes = H[0].long()
        edges = H[1].long()
        N = H.shape[1]

        if N == 0:
            empty_pairs = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_edges = torch.empty((0, 0), dtype=torch.long, device=device)
            return empty_pairs, empty_edges

        num_edges = int(edges.max().item()) + 1
        num_nodes = int(nodes.max().item()) + 1

        # -------------------------------------------------------------------
        # Hyperedge size (order) of every hyperedge.
        # -------------------------------------------------------------------
        sizes = torch.zeros(num_edges, dtype=torch.long, device=device)
        sizes.scatter_add_(0, edges, torch.ones_like(edges))

        # -------------------------------------------------------------------
        # (A) Candidate directed pairs come only from second-order hyperedges.
        # -------------------------------------------------------------------
        is_second_order = sizes[edges] == 2
        so_nodes = nodes[is_second_order]
        so_edges = edges[is_second_order]

        if so_nodes.numel() == 0:
            empty_pairs = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_edges = torch.empty((0, 0), dtype=torch.long, device=device)
            return empty_pairs, empty_edges

        perm = torch.argsort(so_edges, stable=True)
        so_nodes = so_nodes[perm].view(-1, 2)               # M x 2
        a, b = so_nodes[:, 0], so_nodes[:, 1]

        pair_src = torch.cat([a, b])
        pair_dst = torch.cat([b, a])
        pair_stack = torch.stack([pair_src, pair_dst], dim=1)            # (2M) x 2
        unique_pairs = torch.unique(pair_stack, dim=0, sorted=True)      # P x 2
        P = unique_pairs.shape[0]

        # -------------------------------------------------------------------
        # (B) For every hyperedge enumerate all ordered (a, b) pairs (a != b)
        #     of its members and tag them with the hyperedge index.
        # -------------------------------------------------------------------
        perm_h = torch.argsort(edges, stable=True)
        nodes_s = nodes[perm_h]
        edges_s = edges[perm_h]

        offsets = torch.zeros(num_edges + 1, dtype=torch.long, device=device)
        offsets[1:] = sizes.cumsum(0)

        sizes_per_entry = sizes[edges_s]
        rep_count = sizes_per_entry - 1                                  # partners per entry

        left_nodes = nodes_s.repeat_interleave(rep_count)
        left_edges = edges_s.repeat_interleave(rep_count)

        within_pos = torch.arange(N, device=device) - offsets[edges_s]   # position of entry in its edge

        total_out = int(rep_count.sum().item())
        out_starts = torch.zeros(N + 1, dtype=torch.long, device=device)
        out_starts[1:] = rep_count.cumsum(0)

        left_k = torch.arange(N, device=device).repeat_interleave(rep_count)
        j = torch.arange(total_out, device=device) - out_starts[left_k]

        i_per_left = within_pos[left_k]
        right_within = torch.where(j < i_per_left, j, j + 1)
        right_global = offsets[left_edges] + right_within
        right_nodes = nodes_s[right_global]

        # -------------------------------------------------------------------
        # (C) Keep only those (src, dst, edge) triples whose (src, dst) is one
        #     of the unique pairs derived from second-order hyperedges.
        # -------------------------------------------------------------------
        stride = num_nodes
        triple_code = left_nodes * stride + right_nodes
        unique_code = unique_pairs[:, 0] * stride + unique_pairs[:, 1]   # already sorted by torch.unique

        pos = torch.searchsorted(unique_code, triple_code)
        pos_clamped = pos.clamp(max=P - 1)
        found = unique_code[pos_clamped] == triple_code

        valid_pair_idx = pos_clamped[found]
        valid_edge = left_edges[found]

        # -------------------------------------------------------------------
        # (D) Bucket the edges by pair index into a P x K tensor padded with -1.
        # -------------------------------------------------------------------
        edge_counts = torch.zeros(P, dtype=torch.long, device=device)
        edge_counts.scatter_add_(0, valid_pair_idx, torch.ones_like(valid_pair_idx))
        K = int(edge_counts.max().item())

        result = torch.full((P, K), -1, dtype=torch.long, device=device)

        sort_perm = torch.argsort(valid_pair_idx, stable=True)
        sorted_pair = valid_pair_idx[sort_perm]
        sorted_edge = valid_edge[sort_perm]

        bucket_starts = torch.zeros(P + 1, dtype=torch.long, device=device)
        bucket_starts[1:] = edge_counts.cumsum(0)
        within_bucket = torch.arange(sorted_pair.numel(), device=device) - bucket_starts[sorted_pair]

        result[sorted_pair, within_bucket] = sorted_edge

        return unique_pairs.t().contiguous(), result

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'edge_dim={self.edge_dim})')
