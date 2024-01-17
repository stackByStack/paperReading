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

