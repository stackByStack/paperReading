# CHEX: CHannel EXploration for CNN Model Compression

<show-structure for="chapter,procedure" depth="3"/>

## Abstract
- As opposed to pruning-only strategy, we propose
  to repeatedly **prune and regrow** the channels throughout the
  training process, which reduces the risk of pruning important channels prematurely.
-  From intra-layer’s
   aspect, we tackle the channel pruning problem via a well-known **column subset selection** (CSS) formulation. 
- From
   interlayer’s aspect, our regrowing stages open a path for
   dynamically **re-allocating the number of channels** _across_
   **all the layers** under a global channel sparsity constraint.

## Introduction
### existing channel pruning methods
- Towards Efficient Model Compression via Learned Global Ranking
  - determining a target model complexity can be difficult for optimizing various **embodied AI applications** such as autonomous robots, drones, and user-facing applications
  - This work takes
    a first step toward making this process more efficient by altering the goal of model compression to producing a **_set_** of
    ConvNets with _**various**_ accuracy and latency **trade-offs** instead of producing one ConvNet targeting some pre-defined
    latency constraint

![image_20240212_132800.png](image_20240212_132800.png)

- Channel pruning
  guided by **classification loss** and **feature importance**

- <a href="Filter-Pruning-via-Geometric-Median-for-Deep-Convolutional-Neural-Networks-Acceleration.md"></a>

- Channel pruning for accelerating very deep neural networks
  - Sampler Based
  - Minimized Reconstruction Error of sampled ones.
![image_20240212_170800.png](image_20240212_170800.png)

Now 30


















