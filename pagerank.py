# pagerank

import torch
import dgl
import dgl.function as fn
from dgl.data import CoraGraphDataset


# N = 100
# g = nx.erdos_renyi_graph(N, 0.05)
# g = dgl.DGLGraph(g)


def compute_pagerank(g, N, DAMP, K):
    g.ndata['pv'] = torch.ones(N) / N
    degrees = g.out_degrees(g.nodes()).type(torch.float32)
    for k in range(K):
        g.ndata['pv'] = g.ndata['pv'] / degrees
        g.update_all(message_func=fn.copy_src(src='pv', out='m'),
                     reduce_func=fn.sum(msg='m', out='pv'))
        g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['pv']
    return g.ndata['pv']

def topk_idx(data, N, DAMP, K, k):
    pr = compute_pagerank(data, N, DAMP, K)
    # print(pr)
    topk_pr = torch.topk(pr, k)
    # print(topk_pr)
    # print(topk_pr[0].size())
    return topk_pr

# cora = CoraGraphDataset()
# cora = cora[0]
# N = cora.num_nodes()
# DAMP = 0.85
# K = 10
# k = 5
# topk_hop_idx(cora, N, DAMP, K, k)