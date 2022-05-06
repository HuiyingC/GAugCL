
# CAugCL-kr

## Installation Prerequisites

GAugCL-kr needs the following packages to be installed beforehand:

* Python 3.9+
* PyTorch 1.11.0
* torch-scatter
* torch-sparse
* PyTorch-Geometric 2.0.4
* DGL 0.7+
* Scikit-learn 0.24+
* Numpy
* tqdm
* NetworkX
* Sphinx==4.0.2
* myst-parser==0.15.2
* sphinx-rtd-theme==1.0.0
* sphinx-autodoc-typehints==1.12.0
* livereload==2.6.3


## For Dr. Hamdi:
To avoid any environment conflict issues, which bring me lots of headache, and for your grading convenience, I made a Jupyter Notebook with running experiments on Cora dataset. You can refer to [examples_cora.ipynb](examples_cora.ipynb).


# Execution
`python train.py <dataset> <augmentor>`

`<dataset> in ['Cora', 'Citeseer']`

`<augmentor> in [baseline', 'baseline-noFM', 'baseline-noND',
                  'FM+RW', 'FM+DF', 'FM+TopK', 'FM+Khop',
                  'FM+RW+TopK', 'FM+RW+Khop']`

eg. `python train.py Cora baseline`

To avoid UserWarning: resource_tracker Warning, add `-W ignore`

There is a bit hard coding. If you want to use Citeseer dataset, please go to [augmentors/khop_sub.py](augmentors/khop_sub.py), and uncomment below lines: 
```python
pr_topk = topk_idx(citeseer, self.N, DAMP=0.85, K=100, k=self.k)
trick = torch.tensor([3326])
citeseer = CiteseerGraphDataset()[0]
```

Also, in [augmentors/topk_sub.py](augmentors/topk_sub.py), uncomment:
```python
pr_topk = topk_idx(citeseer, self.N, DAMP=0.85, K=100, k=self.k)
citeseer = CiteseerGraphDataset()[0]
```


# Datasets Used and Example Results
* Cora
* CiteSeer
* See .log files in [logs/*](logs/)



# Overview

The GAugCL-kr model implements four main components of graph contrastive learning algorithms:

* Graph augmentation: transforms input graphs into congruent graph views.
* Contrasting architectures and modes: generate positive and negative pairs according to node and graph embeddings.
* Contrastive objectives: computes the likelihood score for positive and negative pairs.
* Negative mining strategies: improves the negative sample set by considering the relative similarity (the hardness) of negative sample.

The model also implements utilities for training models, evaluating model performance, and managing experiments.


## Graph Augmentation

In [augmentors](augmentors), GAugCL-kr provides the `Augmentor` base class, which offers a universal interface for graph augmentation functions. Specifically, GAugCL-kr implements the following augmentation functions:

| Augmentation                            | Class name       |
|-----------------------------------------|------------------|
| Edge Adding (EA)                        | `EdgeAdding`     |
| Edge Removing (ER)                      | `EdgeRemoving`   |
| Feature Masking (FM)                    | `FeatureMasking` |
| Personalized PageRank (PPR)             | `PPRDiffusion`   |
| Topk Subgraph Sampling (Khop)           | `TopkSubgraph`   |
| Khop Subgraph Sampling (Khop)           | `KhopSubgraph`   |
| Node Dropping (ND)                      | `NodeDropping`   |
| Subgraphs induced by Random Walks (RWS) | `RWSampling`     |

Call these augmentation functions by feeding with a `Graph` in a tuple form of node features, edge index, and edge features `(x, edge_index, edge_attrs)` will produce corresponding augmented graphs.

To compose a list of augmentation instances `augmentors`, you need to use the `Compose` class:
```python
import augmentors as A

aug = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
```
You can also use the RandomChoice class to randomly draw a few augmentations each time:
```python
import augmentors as A

aug = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=200),
                          A.TopKSubgraph(N, k=10),
                          A.FeatureMasking(pf=0.3)])
```


## Contrastive Objectives

In [losses](losses), GAugCL-kr implements the following contrastive objectives:

| Contrastive objectives               | Class name        |
| ------------------------------------ | ----------------- |
| InfoNCE loss                         | `InfoNCE`         |
| Jensen-Shannon Divergence (JSD) loss | `JSD`             |


## Utilities

[Evaluator functions](eval) to evaluate the embedding quality:

| Evaluator              | Class name     |
| ---------------------- | -------------- |
| Logistic regression    | `LREvaluator`  |
| Support vector machine | `SVMEvaluator` |
| Random forest          | `RFEvaluator`  |


# Baselines
[baselines](baselines) to compare results with state-art-of prior works:

| Baseline  | File            | Paper                                                                                                                                                    |
|-----------|-----------------|------------------------|
| GRACE     | `GRACE.py`      | [ICML 2020_Deep Graph Contrastive Representation Learning](https://arxiv.org/abs/2006.04131)    |
| GraphCL   | `GraphCL.py`    | [NeurIPS 2020_Graph Contrastive Learning with Augmentations](https://arxiv.org/abs/2010.13902)   |
| InfoGraph | `InfoGraph.py`  | [NeurIPS 2021_InfoGCL: Information-Aware Graph Contrastive Learning](https://proceedings.neurips.cc/paper/2021/file/ff1e68e74c6b16a1a7b5d958b95e120c-Paper.pdf) |
| MVGRL     | `MVGRL_node.py` | [ICML 2020_Contrastive Multi-View Representation Learning on Graphs](https://arxiv.org/abs/2006.05582)    |