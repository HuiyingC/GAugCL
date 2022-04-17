import torch
import networkx as nx
import torch.nn.functional as F

from typing import Optional
from utils import normalize
from torch_sparse import SparseTensor, coalesce
from torch_scatter import scatter
from torch_geometric.transforms import GDC
from torch.distributions import Uniform, Beta
from torch_geometric.utils import dropout_adj, to_networkx, to_undirected, degree, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix, sort_edge_index, add_self_loops, subgraph
from torch.distributions.bernoulli import Bernoulli




def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[:, drop_mask] = 0

    return x


# def dropout_feature(x: torch.FloatTensor, drop_prob: float) -> torch.FloatTensor:
#     return F.dropout(x, p=1. - drop_prob)


def get_feature_weights(x, centrality, sparse=True):
    if sparse:
        x = x.to(torch.bool).to(torch.float32)
    else:
        x = x.abs()
    w = x.t() @ centrality
    w = w.log()

    return normalize(w)


def drop_feature_by_weight(x, weights, drop_prob: float, threshold: float = 0.7):
    weights = weights / weights.mean() * drop_prob
    weights = weights.where(weights < threshold, torch.ones_like(weights) * threshold)  # clip
    drop_mask = torch.bernoulli(weights).to(torch.bool)
    x = x.clone()
    x[:, drop_mask] = 0.
    return x


def get_eigenvector_weights(data, nx_G):
    N = data.edge_index.max().item() + 1
    def _eigenvector_centrality(nx_G, N):
        # graph = to_networkx(data)
        graph = nx_G
        x = nx.eigenvector_centrality(graph)
        x = [x[i] for i in range(N)]
        return torch.tensor(x, dtype=torch.float32)
            # .to(data.edge_index.device)

    evc = _eigenvector_centrality(nx_G, N)
    scaled_evc = evc.where(evc > 0, torch.zeros_like(evc))
    scaled_evc = scaled_evc + 1e-8
    s = scaled_evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]

    return normalize(s_col), evc


def get_pagerank_weights(data, aggr: str = 'sink', k: int = 10):
    def _compute_pagerank(edge_index, damp: float = 0.85, k: int = 10):
        num_nodes = edge_index.max().item() + 1
        deg_out = degree(edge_index[0])
        x = torch.ones((num_nodes,)).to(edge_index.device).to(torch.float32)

        for i in range(k):
            edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
            agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

            x = (1 - damp) * x + damp * agg_msg

        return x

    pv = _compute_pagerank(data.edge_index, k=k)
    pv_row = pv[data.edge_index[0]].to(torch.float32)
    pv_col = pv[data.edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col

    return normalize(s), pv


def drop_edge_by_weight(edge_index, weights, drop_prob: float, threshold: float = 0.7):
    weights = weights / weights.mean() * drop_prob
    weights = weights.where(weights < threshold, torch.ones_like(weights) * threshold)
    drop_mask = torch.bernoulli(1. - weights).to(torch.bool)
    # print('drop_mask', drop_mask)
    return edge_index[:, drop_mask]


def get_subgraph(x, edge_index, idx):
    adj = to_scipy_sparse_matrix(edge_index).tocsr()
    x_sampled = x[idx]
    edge_index_sampled = from_scipy_sparse_matrix(adj[idx, :][:, idx])
    return x_sampled, edge_index_sampled


def sample_nodes(x, edge_index, sample_size):
    idx = torch.randperm(x.size(0))[:sample_size]
    return get_subgraph(x, edge_index, idx), idx


def compute_ppr(edge_index, edge_weight=None, alpha=0.2, eps=0.1, ignore_edge_attr=True, add_self_loop=True):
    N = edge_index.max().item() + 1
    if ignore_edge_attr or edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), device=edge_index.device)
    if add_self_loop:
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=N)
        edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, N, normalization='sym')
    diff_mat = GDC().diffusion_matrix_exact(
        edge_index, edge_weight, N, method='ppr', alpha=alpha)
    edge_index, edge_weight = GDC().sparsify_dense(diff_mat, method='threshold', eps=eps)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, N, normalization='sym')

    return edge_index, edge_weight


def get_sparse_adj(edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,
                   add_self_loop: bool = True) -> torch.sparse.Tensor:
    num_nodes = edge_index.max().item() + 1
    num_edges = edge_index.size(1)

    if edge_weight is None:
        edge_weight = torch.ones((num_edges,), dtype=torch.float32, device=edge_index.device)

    if add_self_loop:
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=num_nodes)
        edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes, num_nodes)

    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, num_nodes, normalization='sym')

    adj_t = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes)).coalesce()

    return adj_t.t()


def coalesce_edge_index(edge_index: torch.Tensor, edge_weights: Optional[torch.Tensor] = None) -> (torch.Tensor, torch.FloatTensor):
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    edge_weights = edge_weights if edge_weights is not None else torch.ones((num_edges,), dtype=torch.float32, device=edge_index.device)

    return coalesce(edge_index, edge_weights, m=num_nodes, n=num_nodes)


def add_edge(edge_index: torch.Tensor, ratio: float) -> torch.Tensor:
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    num_add = int(num_edges * ratio)

    new_edge_index = torch.randint(0, num_nodes - 1, size=(2, num_add)).to(edge_index.device)
    edge_index = torch.cat([edge_index, new_edge_index], dim=1)

    edge_index = sort_edge_index(edge_index)[0]

    return coalesce_edge_index(edge_index)[0]


def drop_node(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, keep_prob: float = 0.5) -> (torch.Tensor, Optional[torch.Tensor]):
    num_nodes = edge_index.max().item() + 1
    probs = torch.tensor([keep_prob for _ in range(num_nodes)])
    dist = Bernoulli(probs)

    subset = dist.sample().to(torch.bool).to(edge_index.device)
    edge_index, edge_weight = subgraph(subset, edge_index, edge_weight)

    return edge_index, edge_weight


def random_walk_subgraph(edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None, batch_size: int = 1000, length: int = 10):
    num_nodes = edge_index.max().item() + 1

    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))

    start = torch.randint(0, num_nodes, size=(batch_size, ), dtype=torch.long).to(edge_index.device)
    node_idx = adj.random_walk(start.flatten(), length).view(-1)

    edge_index, edge_weight = subgraph(node_idx, edge_index, edge_weight)

    return edge_index, edge_weight
