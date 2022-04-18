import os
import sys

import torch
import os.path as osp
import losses as L
import augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam
from eval import get_split, LREvaluator
from models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid



class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    opt = {}
    opt['data'] = ['Cora', 'CiteSeer', 'PubMed']
    opt['aug'] = ['baseline', 'baseline-noFM', 'baseline-noND',
                  'FM+RW', 'FM+DF', 'FM+TopK', 'FM+Khop',
                  'TopK+RW', 'RW+Khop', 'TopK+Khop',
                  'FM+TopK+RW', 'FM+RW+Khop', 'FM+TopK+Khop']
    opt['pe'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    opt['pn'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.7]
    opt['alpha'] = [0.01, 0.05, 0.1, 0.2]
    opt['walk_length_ratio'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    opt['k_ratio'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    opt['num_hops'] = [1, 2, 3, 4, 5]

    data_arg = sys.argv[1]
    # print('dataset:', dataset)
    assert data_arg in opt['data']
    aug = sys.argv[2]
    assert aug in opt['aug']

    # device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = Planetoid(path, name=data_arg, transform=T.NormalizeFeatures())
    data = dataset[0]
    N = data.num_nodes
    # N = data.edge_index.max().item() + 1
    # print("N =", N)
    # .to(device)

    # store outputs into a log file
    # if os.path.exists(f"{dataset}_{aug}.log"):
    #     os.remove(f"{dataset}_{aug}.log")
    log = open(f"logs/{data_arg}_{aug}.log", "a")
    sys.stdout = log

    def run_model(aug1, aug2):
        gconv = GConv(input_dim=dataset.num_features, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2)
        # .to(device)
        encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=32, proj_dim=32)
        # .to(device)
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True)
        # .to(device)

        optimizer = Adam(encoder_model.parameters(), lr=0.01)

        with tqdm(total=200, desc='(T)') as pbar:
            for epoch in range(1, 201):
                loss = train(encoder_model, contrast_model, data, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()

        test_result = test(encoder_model, data)
        print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    if aug == 'baseline':
        print('Here are baseline results.')
        for pe in opt['pe']:
            for pn in opt['pn']:
                print(f'aug=[A.EdgeRemoving(pe={pe}), A.NodeDropping(pn={pn}), A.FeatureMasking(pf=0.3)]')
                print('====================================================================================')
                aug1 = A.Compose([A.EdgeRemoving(pe=pe), A.NodeDropping(pn=pn), A.FeatureMasking(pf=0.3)])
                aug2 = A.Compose([A.EdgeRemoving(pe=pe), A.NodeDropping(pn=pn), A.FeatureMasking(pf=0.3)])
                run_model(aug1, aug2)

    if aug == 'baseline-noFM':
        print('Here are baseline-noFM results.')
        for pe in opt['pe']:
            for pn in opt['pn']:
                print(f'aug=[A.EdgeRemoving(pe={pe}), A.NodeDropping(pn={pn})]')
                print('====================================================================================')
                aug1 = A.Compose([A.EdgeRemoving(pe=pe), A.NodeDropping(pn=pn)])
                aug2 = A.Compose([A.EdgeRemoving(pe=pe), A.NodeDropping(pn=pn)])
                run_model(aug1, aug2)

    if aug == 'baseline-noND':
        print('Here are baseline-noND results.')
        for pe in opt['pe']:
            print(f'aug=[A.EdgeRemoving(pe={pe}), A.FeatureMasking(pf=0.3)]')
            print('====================================================================================')
            aug1 = A.Compose([A.EdgeRemoving(pe=pe), A.FeatureMasking(pf=0.3)])
            aug2 = A.Compose([A.EdgeRemoving(pe=pe), A.FeatureMasking(pf=0.3)])
            run_model(aug1, aug2)

    if aug == 'FM+RW':
        for wl_r in opt['walk_length_ratio']:
            wl = round(N * wl_r)
            print(f'=================================================================')
            print(f'walk_length_ratio={wl_r}, RWSampling_walk_length={wl}')
            aug1 = A.Compose([A.RWSampling(num_seeds=1000, walk_length=wl),
                              A.FeatureMasking(pf=0.3)])
            aug2 = A.Compose([A.RWSampling(num_seeds=1000, walk_length=wl),
                              A.FeatureMasking(pf=0.3)])
            run_model(aug1, aug2)

    if aug == 'FM+DF':
        for a in opt['alpha']:
            print(f'=================================================================')
            print(f'PPRDiffusion_alpha={a}')
            aug1 = A.Compose([A.FeatureMasking(pf=0.3),
                              A.PPRDiffusion(alpha=a)])
            aug2 = A.Compose([A.FeatureMasking(pf=0.3),
                              A.PPRDiffusion(alpha=a)])
            run_model(aug1, aug2)

    if aug == 'FM+TopK':
        for r in opt['k_ratio']:
            k = round(N * r)
            print(f'=================================================================')
            print(f'k_ratio={r}, TopKSubgraph_k={k}')
            aug1 = A.Compose([A.FeatureMasking(pf=0.3),
                              # A.TopKSubgraph(N, k=k, data=data_arg)])
                              A.TopKSubgraph(N, k=k)])
            aug2 = A.Compose([A.FeatureMasking(pf=0.3),
                              A.TopKSubgraph(N, k=k)])
            run_model(aug1, aug2)

    if aug == 'FM+Khop':
        for r in opt['k_ratio']:
            k = round(N * r)
            for num_hop in opt['num_hops']:
                print(f'=================================================================')
                print(f'k_ratio={r}, k={k}, KhopSubgraph_num_hops={num_hop}')
                aug1 = A.Compose([A.FeatureMasking(pf=0.3),
                                  A.KhopSubgraph(N, k=k, num_hops=num_hop)])
                aug2 = A.Compose([A.FeatureMasking(pf=0.3),
                                  A.KhopSubgraph(N, k=k, num_hops=num_hop)])
                run_model(aug1, aug2)

    if aug == 'TopK+RW':
        for r in opt['k_ratio']:
            k = round(N * r)
            for wl_r in opt['walk_length_ratio']:
                wl = round(N * wl_r)
                print(f'=================================================================')
                print(f'k_ratio={r}, TopKSubgraph_k={k}, walk_length_ratio={wl_r}, RWSampling_walk_length={wl}')
                aug1 = A.Compose([A.RWSampling(num_seeds=1000, walk_length=wl),
                                  A.TopKSubgraph(N, k=k)])
                aug2 = A.Compose([A.RWSampling(num_seeds=1000, walk_length=wl),
                                  A.TopKSubgraph(N, k=k)])
                run_model(aug1, aug2)

    if aug == 'FM+TopK+RW':
        for r in opt['k_ratio']:
            k = round(N * r)
            for wl_r in opt['walk_length_ratio']:
                wl = round(N * wl_r)
                print(f'=================================================================')
                print(f'k_ratio={r}, TopKSubgraph_k={k}, walk_length_ratio={wl_r}, RWSampling_walk_length={wl}')
                aug1 = A.Compose([A.RWSampling(num_seeds=1000, walk_length=wl),
                                  A.TopKSubgraph(N, k=k),
                                  A.FeatureMasking(pf=0.3)])
                aug2 = A.Compose([A.RWSampling(num_seeds=1000, walk_length=wl),
                                  A.TopKSubgraph(N, k=k),
                                  A.FeatureMasking(pf=0.3)])
                run_model(aug1, aug2)

    if aug == 'FM+RW+Khop':
        for r in opt['k_ratio']:
            k = round(N * r)
            for num_hop in opt['num_hops']:
                # for wl_r in opt['walk_length_ratio']:
                for wl_r in [0.8]:
                    wl = round(N * wl_r)
                    print(f'=================================================================')
                    print(f'k_ratio={r}, k={k}, KhopSubgraph_num_hops={num_hop}, walk_length_ratio={wl_r}, RWSampling_walk_length={wl}')
                    aug1 = A.Compose([A.RWSampling(num_seeds=1000, walk_length=wl),
                                      A.KhopSubgraph(N, k=k, num_hops=num_hop),
                                      A.FeatureMasking(pf=0.3)])
                    aug2 = A.Compose([A.RWSampling(num_seeds=1000, walk_length=wl),
                                      A.KhopSubgraph(N, k=k, num_hops=num_hop),
                                      A.FeatureMasking(pf=0.3)])
                    run_model(aug1, aug2)

    if aug == 'FM+TopK+Khop':
        for r in opt['k_ratio']:
            k = round(N * r)
            for num_hop in opt['num_hops']:
                print(f'=================================================================')
                print(f'k_ratio={r}, KhopSubgraph_k={k}, KhopSubgraph_num_hops={num_hop}')
                aug1 = A.Compose([A.TopKSubgraph(N, k=k),
                                  A.FeatureMasking(pf=0.3),
                                  A.KhopSubgraph(N, k=k, num_hops=num_hop)])
                aug2 = A.Compose([A.TopKSubgraph(N, k=k),
                                  A.FeatureMasking(pf=0.3),
                                  A.KhopSubgraph(N, k=k, num_hops=num_hop)])
                run_model(aug1, aug2)


if __name__ == '__main__':
    main()
    # print(torch.__version__)