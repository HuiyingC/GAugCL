
# Install

## Prerequisites

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

# Execution
`python train.py <dataset> <augmentor>`

eg. `python train.py Cora baseline`

To avoid UserWarning: resource_tracker Warning, add `-W ignore`

# Datasets used and example results
* Cora
* CiteSeer
* See .log files in `logs/`


# Overview

The GAugCL-kr model implements four main components of graph contrastive learning algorithms:

* Graph augmentation: transforms input graphs into congruent graph views.
* Contrasting architectures and modes: generate positive and negative pairs according to node and graph embeddings.
* Contrastive objectives: computes the likelihood score for positive and negative pairs.
* Negative mining strategies: improves the negative sample set by considering the relative similarity (the hardness) of negative sample.

The model also implements utilities for training models, evaluating model performance, and managing experiments.


## Graph Augmentation

In `GCL.augmentors`, GAugCL-kr provides the `Augmentor` base class, which offers a universal interface for graph augmentation functions. Specifically, PyGCL implements the following augmentation functions:

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


## Contrastive Objectives

In `losses`, GAugCL-kr implements the following contrastive objectives:

| Contrastive objectives               | Class name        |
| ------------------------------------ | ----------------- |
| InfoNCE loss                         | `InfoNCE`         |
| Jensen-Shannon Divergence (JSD) loss | `JSD`             |


## Utilities

Evaluator functions to evaluate the embedding quality:

| Evaluator              | Class name     |
| ---------------------- | -------------- |
| Logistic regression    | `LREvaluator`  |
| Support vector machine | `SVMEvaluator` |
| Random forest          | `RFEvaluator`  |

