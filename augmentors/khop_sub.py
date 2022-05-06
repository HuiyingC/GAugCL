import torch
from augmentors.augmentor import Graph, Augmentor
# from torch_geometric.utils import subgraph, k_hop_subgraph
from typing import List, Optional, Union
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from pagerank import topk_idx
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset


# updated from torch_geometric.utils
def subgraph(subset: Union[Tensor, List[int]], edge_index: Tensor,
             edge_attr: Optional[Tensor] = None, relabel_nodes: bool = False,
             num_nodes: Optional[int] = None, return_edge_mask: bool = False):
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    device = edge_index.device
    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        node_mask = subset
        num_nodes = node_mask.size(0)
        if relabel_nodes:
            node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
                                   device=device)
            node_idx[subset] = torch.arange(subset.sum().item(), device=device)
    else:
        # num_nodes = maybe_num_nodes(edge_index, num_nodes)   # fix out of range bug in aug == 'FM+TopK+Khop'
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        node_mask[subset] = 1

        if relabel_nodes:
            node_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
            node_idx[subset] = torch.arange(subset.size(0), device=device)

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index = node_idx[edge_index]

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr

def k_hop_subgraph(edge_index, node_idx, num_hops: int = 3, relabel_nodes=False, num_nodes=None,
                   flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """
    # num_nodes = maybe_num_nodes(edge_index, num_nodes)
    num_nodes = torch.max(node_idx) + 1    # fix out of range bug in aug == 'FM+TopK+Khop'
    # print('num_nodes', num_nodes)
    # print('max in node_idx', torch.max(node_idx))

    # print('edge_index', edge_index)
    # print('edge_index.size', edge_index.size())
    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index
    # print('edge_index max node_idx:', edge_index.max().item() + 1)

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        # print(f'node_mask.size(): {node_mask.size()}, row.size(): {row.size()}')

        # IndexError: index out of range in self
        # https://medium.com/emulation-nerd/every-index-based-operation-youll-ever-need-in-pytorch-a7cef65ea94c
        # https://jovian.ai/richardso21/01-tensor-operations#C2
        # print('row max node_idx', max(row))
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


# idea: get node influence (pagerank) => compute topk node idx => get khop subgraph of topk node idx
# question to consider: overlap respective fields
class KhopSubgraph(Augmentor):
    # def __init__(self, N: int, k: int, data):
    def __init__(self, N: int, k: int, num_hops: int):
        super(KhopSubgraph, self).__init__()
        self.N = N
        self.k = k
        self.num_hops = num_hops

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()

        pr_topk = topk_idx(cora, self.N, DAMP=0.85, K=100, k=self.k)
        # pr_topk = topk_idx(citeseer, self.N, DAMP=0.85, K=100, k=self.k)
        # pr_topk = topk_idx(pumb, self.N, DAMP=0.85, K=100, k=self.k)
        topk_subset = pr_topk[1]
        # print('topk_subset', topk_subset)

        # trick for citeseer dataset error: IndexError: index out of range in self
        trick = torch.tensor([2707])
        # trick = torch.tensor([3326])
        # trick = torch.tensor([19716])

        topk_subset = torch.cat((topk_subset,trick))

        # extract khop for every node_idx in topk subset
        khop_subset, _, _, _ = k_hop_subgraph(edge_index, topk_subset, self.num_hops)

        num_nodes = torch.max(khop_subset) + 1  # fix out of range bug in aug == 'FM+TopK+Khop'
        edge_index, edge_weight = subgraph(khop_subset, edge_index, edge_weights, num_nodes=num_nodes)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


cora = CoraGraphDataset()[0]
# citeseer = CiteseerGraphDataset()[0]
# pumb = PubmedGraphDataset()[0]


