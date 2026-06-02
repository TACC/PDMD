import torch
from torch_geometric.data import Data

class HyperGraphData(Data):
    r"""A data object describing a hypergraph, extending :class:`torch_geometric.data.Data`.

    A hyperedge connects two or more nodes. In this representation, the
    :obj:`hyperedge_index` is a tensor of shape :obj:`[2, num_connections]`,
    where the first row contains node indices and the second row contains
    hyperedge indices. Each (node, hyperedge) pair indicates that the node
    belongs to that hyperedge.

    Args:
        x (Tensor, optional): Node feature matrix of shape
            :obj:`[num_nodes, num_node_features]`.
        hyperedge_index (LongTensor, optional): Hyperedge connectivity in COO
            format of shape :obj:`[2, num_connections]`. Row 0 contains node
            indices, row 1 contains hyperedge indices.
        hyperedge_attr (Tensor, optional): Hyperedge feature matrix of shape
            :obj:`[num_hyperedges, num_hyperedge_features]`.
        y (Tensor, optional): Graph-level or node-level targets.
        num_hyperedges (int, optional): Number of hyperedges. If not given,
            inferred from :obj:`hyperedge_index`.
        **kwargs: Additional attributes.

    Example:
        >>> # A hypergraph with 4 nodes and 2 hyperedges:
        >>> #   e0 connects nodes {0, 1, 2}
        >>> #   e1 connects nodes {1, 2, 3}
        >>> x = torch.randn(4, 8)
        >>> hyperedge_index = torch.tensor([
        ...     [0, 1, 2, 1, 2, 3],   # node indices
        ...     [0, 0, 0, 1, 1, 1],   # hyperedge indices
        ... ], dtype=torch.long)
        >>> hyperedge_attr = torch.randn(2, 16)
        >>> data = HyperedgeData(
        ...     x=x,
        ...     hyperedge_index=hyperedge_index,
        ...     hyperedge_attr=hyperedge_attr,
        ... )
    """

    def __init__(
        self,
        x=None,
        hyperedge_index=None,
        hyperedge_attr=None,
        y=None,
        num_hyperedges=None,
        **kwargs,
    ):
        super().__init__(
            x=x,
            hyperedge_index=hyperedge_index,
            hyperedge_attr=hyperedge_attr,
            y=y,
            **kwargs,
        )
        if num_hyperedges is not None:
            self.num_hyperedges = num_hyperedges

    @property
    def num_hyperedges(self) -> int:
        """Number of hyperedges in the hypergraph."""
        if "num_hyperedges" in self._store:
            return self._store["num_hyperedges"]
        if self.hyperedge_attr is not None:
            return self.hyperedge_attr.size(0)
        if self.hyperedge_index is not None and self.hyperedge_index.numel() > 0:
            return int(self.hyperedge_index[1].max()) + 1
        return 0

    @num_hyperedges.setter
    def num_hyperedges(self, value: int):
        self._store["num_hyperedges"] = value

    def __inc__(self, key, value, *args, **kwargs):
        # When batching multiple hypergraphs, increment the node-index row
        # (row 0) by num_nodes and the hyperedge-index row (row 1) by
        # num_hyperedges so indices don't collide across examples.
        if key == "hyperedge_index":
            return torch.tensor([[self.num_nodes], [self.num_hyperedges]])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # hyperedge_index is concatenated along the last dim (like edge_index).
        # hyperedge_attr is concatenated along dim 0 (like node features).
        if key == "hyperedge_index":
            return -1
        if key == "hyperedge_attr":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def is_valid(self) -> bool:
        """Sanity-check the consistency of stored tensors."""
        if self.hyperedge_index is None:
            return True
        if self.hyperedge_index.dim() != 2 or self.hyperedge_index.size(0) != 2:
            return False
        if self.hyperedge_attr is not None:
            if self.hyperedge_attr.size(0) != self.num_hyperedges:
                return False
        return True
