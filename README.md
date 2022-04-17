
# Install

## Prerequisites

PyGCL needs the following packages to be installed beforehand:

* Python 3.8+
* PyTorch 1.9+
* PyTorch-Geometric 1.7
* DGL 0.7+
* Scikit-learn 0.24+
* Numpy
* tqdm
* NetworkX

# Execution
`python GRACE.py <dataset> <augmentor>`

eg. `python train.py Cora baseline`

To avoid UserWarning: resource_tracker Warning, add `-W ignore`

# Datasets used and example results
* Cora
* CiteSeer
* PubMed
* See .log files in `logs/`


# Overview

The GAugCL model implements four main components of graph contrastive learning algorithms:

* Graph augmentation: transforms input graphs into congruent graph views.
* Contrasting architectures and modes: generate positive and negative pairs according to node and graph embeddings.
* Contrastive objectives: computes the likelihood score for positive and negative pairs.
* Negative mining strategies: improves the negative sample set by considering the relative similarity (the hardness) of negative sample.

The model also implements utilities for training models, evaluating model performance, and managing experiments.


## Graph Augmentation

In `GCL.augmentors`, PyGCL provides the `Augmentor` base class, which offers a universal interface for graph augmentation functions. Specifically, PyGCL implements the following augmentation functions:

| Augmentation                            | Class name        |
|-----------------------------------------| ----------------- |
| Edge Adding (EA)                        | `EdgeAdding`      |
| Edge Removing (ER)                      | `EdgeRemoving`    |
| Feature Masking (FM)                    | `FeatureMasking`  |
| Personalized PageRank (PPR)             | `PPRDiffusion`    |
| Khop Subgraph Sampling (Khop)           | `KhopSubgraph`    |
| Node Dropping (ND)                      | `NodeDropping`    |
| Subgraphs induced by Random Walks (RWS) | `RWSampling`      |
| Ego-net Sampling (ES)                   | `Identity`        |

Call these augmentation functions by feeding with a `Graph` in a tuple form of node features, edge index, and edge features `(x, edge_index, edge_attrs)` will produce corresponding augmented graphs.

To compose a list of augmentation instances `augmentors`, you need to use the `Compose` class:

```python
import augmentors as A

aug = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
```



===========================================

Internally, PyGCL calls `Sampler` classes in `GCL.models` that receive embeddings and produce positive/negative masks. PyGCL implements three contrasting modes: (a) Local-Local (L2L), (b) Global-Global (G2G), and (c) Global-Local (G2L) modes. L2L and G2G modes contrast embeddings at the same scale and the latter G2L one performs cross-scale contrasting. To implement your own GCL model, you may also use these provided sampler models:

| Contrastive modes                    | Class name          |
| ------------------------------------ | ------------------- |
| Same-scale contrasting (L2L and G2G) | `SameScaleSampler`  |
| Cross-scale contrasting (G2L)        | `CrossScaleSampler` |

* For L2L and G2G, embedding pairs of the same node/graph in different views constitute positive pairs. You can refer to [GRACE](examples/GRACE.py) and [GraphCL](examples/GraphCL.py) for examples.
* For G2L, node-graph embedding pairs form positives. Note that for single-graph datasets, the G2L mode requires explicit negative sampling (otherwise no negatives for contrasting). You can refer to [DGI](examples/DGI_transductive.py) for an example.
* Some models (e.g., GRACE) add extra intra-view negative samples. You may manually call `sampler.add_intraview_negs` to enlarge the negative sample set.
* Note that the bootstrapping latent model involves some special model design (asymmetric online/offline encoders and momentum weight updates). You may refer to [BGRL](examples/BGRL.py) for details.


## Contrastive Objectives

In `losses`, PyGCL implements the following contrastive objectives:

| Contrastive objectives               | Class name        |
| ------------------------------------ | ----------------- |
| InfoNCE loss                         | `InfoNCE`         |
| Jensen-Shannon Divergence (JSD) loss | `JSD`             |

===============================================


## Utilities

Evaluator functions to evaluate the embedding quality:

| Evaluator              | Class name     |
| ---------------------- | -------------- |
| Logistic regression    | `LREvaluator`  |
| Support vector machine | `SVMEvaluator` |
| Random forest          | `RFEvaluator`  |


=================================================
## Findings * explaination
* Droping/removing ratio increases the accuracy in Cora -> Cora is super sparse