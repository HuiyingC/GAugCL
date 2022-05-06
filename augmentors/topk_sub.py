import torch
from augmentors.augmentor import Graph, Augmentor
from torch_geometric.utils import subgraph
from pagerank import topk_idx
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset



# idea: compute node influence (by pagerank) => get topk most influential node idx => get subgraph
class TopKSubgraph(Augmentor):
    # def __init__(self, N: int, k: int, data):
    def __init__(self, N: int, k: int):
        super(TopKSubgraph, self).__init__()
        self.N = N
        self.k = k
        # self.data = data

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        # print('x:', x)
        # N = edge_index.max().item() + 1

        # if self.data == 'Cora': data = CoraGraphDataset()
        # elif self.data == 'CiteSeer': data = CiteseerGraphDataset()
        # else: data = PubmedGraphDataset()
        # data = data[0]

        # compute node influence by pagerank and extract topk subgraph
        pr_topk = topk_idx(cora, self.N, DAMP=0.85, K=100, k=self.k)
        # pr_topk = topk_idx(citeseer, self.N, DAMP=0.85, K=100, k=self.k)
        # pr_topk = topk_idx(pumb, self.N, DAMP=0.85, K=100, k=self.k)
        subset = pr_topk[1]

        edge_index, edge_weight = subgraph(subset, edge_index, edge_weights)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

cora = CoraGraphDataset()[0]
# citeseer = CiteseerGraphDataset()[0]
# pumb = PubmedGraphDataset()[0]


