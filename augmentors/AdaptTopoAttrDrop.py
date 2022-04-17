from augmentors.augmentor import Graph, Augmentor
from augmentors.functional import dropout_adj, drop_edge_by_weight, get_eigenvector_weights
import os.path as osp
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

# dataset = 'cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
# dataset = Planetoid(path, dataset)
# dataset.transform = T.NormalizeFeatures()
# data = dataset[0]
# nx_G = to_networkx(data, to_undirected=True)
# print(nx_G)


# class AugmentTopologyAttributes(Augmentor):
#     def __init__(self, pe=0.5, pf=0.5):
#         self.pe = pe
#         self.pf = pf
#
#     def __call__(self, x, edge_index):
#         edge_index = dropout_adj(edge_index, p=self.pe)[0]
#         x = drop_feature(x, self.pf)
#         return x, edge_index


class AdaptTopoAttrDrop(Augmentor):
    def __init__(self, pe=0.5, threshold=0.7):
        super(AdaptTopoAttrDrop, self).__init__()
        self.pe = pe
        self.threshold = threshold

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        # _, edge_weights = get_eigenvector_weights(g, nx_G)
        edge_index = drop_edge_by_weight(edge_index, edge_weights, self.pe, self.threshold)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
