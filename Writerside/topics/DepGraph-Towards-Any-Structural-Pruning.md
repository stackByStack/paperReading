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

### Pruning Grouped Parameters


## Method
### Dependency in Neural Networks

- (a) Basic dependency 
- (b) Residual dependency 
- (c) Concatenation dependency  
- (d) Reduction dependency

### Dependence Graph
Find a grouping matrix 
```tex
G \in \mathbb{R}^{L \times L}
```
with

```tex 
G_{ij} = 1
```
indicating that the presence of **dependency** between i-th layer and j-th layer.

In this matrix, G_ij is not only determined by the i-th and j-th layers but also affected by those **intermediate layers** between them.

This recursive process (inferred from w1 ⇔ w2 and w2 ⇔ w3) ultimately ends with a transitive relation, w1 ⇔ w2 ⇔ w3.
In this case, we only need two dependencies to describe the relations in group g.

Thus, it can be compressed into a more compact form with fewer edges while retaining the same information.

Formally, D is constructed such that, for all G_ij = 1, there exists **a path in D between vertex i and j**. Therefore, Gij can be derived by examing the presence of a path between vertices i and j in D.

### network decomposition

> It seems like there are challenges in building a dependency graph at the layer level, particularly due to certain basic layers, such as fully-connected layers, having **different pruning schemes**. The mentioned schemes, like w[k, :] and w[:, k], compress **the dimensions of inputs and outputs**. Additionally, **non-parameterized operations like skip connections** also impact the dependency between layers.

To remedy these issues, we propose a new notation to decompose a network F(x; w) into finer and more basic components, denoted as F = {f1, f2, ..., fL}

where each component f refers to either a **parameterized layer** such as convolution or a **non-parameterized operation** such as residual adding.

### Dependency Modeling

![image_20240118_1618.png](image_20240118_1618.png)

![image_20240118_162100.png](image_20240118_162100.png)

### Group-level Pruning
Specifically, for each parameter w with K prunable dimensions indexed by w[k], we introduce a simple regularization term for sparse training, defined as:

![image_20240118_164100.png](image_20240118_164100.png)

![image_20240118_164200.png](image_20240118_164200.png)

`g` is a parameter group consisting of multiple parameters. `k` is a dimension prunable.

After sparse training, we further use a simple relative score

![image_20240118_171200.png](image_20240118_171200.png)

to identify and remove unimportant parameters.

