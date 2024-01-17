# DepGraph: Towards Any Structural Pruning

## Abstract

<tldr>
In this work, we study a highly-challenging yet barely-explored task, any structural pruning, to tackle general structural pruning of arbitrary architecture like CNNs, RNNs, GNNs and Transformers.
</tldr>

The most prominent obstacle towards this goal lies in the structural coupling, which not only forces different layers to **be pruned simultaneously**, but also expects all removed parameters to **be consistently unimportant**, thereby avoiding structural issues and significant performance degradation after pruning.

## Introduction

### deep neural compression

- Train to find unnecessary connection, prune them and re-train 
<a href="https://arxiv.org/abs/1506.02626"></a>
- Teacher and Student
<a href="https://arxiv.org/abs/1503.02531"></a>
- GNN + Binary Representations + Two Kinds of Aggregators
<a href="https://arxiv.org/abs/2109.12872"></a>
- dataset condensation like Factorization, etc.
<a href="https://arxiv.org/abs/2210.16774"></a>
- CNN Quantization
<a href="https://ieeexplore.ieee.org/document/7780890/"></a>

### pruning schemes

#### structural pruning

- Centripetal SGD; Filters cluster in CNN
<a href="https://arxiv.org/pdf/1904.03837.pdf"></a>

- Filter Pruning
<a href="https://arxiv.org/abs/1608.08710"></a>

#### unstructured pruning

- layer-wise pruning
<a href="https://arxiv.org/abs/1705.07565"></a>
- magnitude-based pruning
<a href="https://arxiv.org/abs/2002.04809"></a>

> The core difference between the two lies in that, 
> structural pruning changes the structure of neural networks by physically removing grouped parameters, 
> 
> while unstructured pruning conducts zeroing on partial weights without modification to the network structure.

> In this paper, we strive for a generic scheme towards any structural pruning, where structural pruning over arbitrary network architectures is executed in an automatic fashion, 
> 
> At the heart of our approach is to estimate the Dependency Graph (DepGraph), which explicitly models the interdependency between paired layers in neural networks.
> 

## Related Work

### Structural and Unstructured Pruning

In practice, unstructured pruning, in particular, is straightforward to implement and inherently adaptable to various networks. 

However, it often necessitates specialized AI **accelerators** or software for model acceleration [15]. 

Conversely, structural pruning improves the inference overhead by physically removing parameters from networks, thereby finding a wider domain of applications [29, 38]. 

In the literature, The design space of pruning algorithms encompasses a range of aspects, including **pruning schemes** [21, 39], **parameter selection** [20, 43, 44], **layer sparsity** [27, 49] and **training techniques** [47, 58].



