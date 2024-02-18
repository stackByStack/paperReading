# ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression

<show-structure for="chapter,procedure" depth="3"/>

## Abstract
- we need to prune filters based on statistics information computed from **its next layer**, not the current layer,
which differentiates ThiNet from existing methods

## Introduction

<a href="https://arxiv.org/pdf/1506.02515.pdf">Fast ConvNets Using Group-wise Brain Damage</a>

![image_20240218_002700.png](image_20240218_002700.png)

- some filter level pruning strategies
  - based on the magnitude of weights.
  - measure the importance of each filter by calculating its absolute weight sum.
  -  if most outputs of some neurons are zero, these activations could be expected to be redundant. Compute
     the Average Percentage of Zeros (APoZ) of each filter as its
     importance score
  - adopt Taylor
    expansion to approximate the influence to loss function induced by removing each filter

Beyond pruning, there are other strategies to obtain
small CNN models.
One popular approach is parameter quantization.
Low-rank approximation is also widely studied.

## ThiNet

![image_202402188_130000.png](image_202402188_130000.png)

### Framework of ThiNet

We summarize our
framework as follows:

1. Filter selection
   - The key idea is:
        if we can use a subset of channels in layer (i + 1) ’s **input** to **_approximate_** the **output** in layer i + 1,
        the other channels can be safely removed from the input
        of layer i + 1. 
   - Note that one channel in layer (i + 1) ’s
      input is produced by one filter in layer i, hence we can
      safely prune the corresponding filter in layer i.
2. Pruning
3. Fine-tuning
4. Iterate to step 1 to prune the next layer

### Data-driven channel selection

#### Collecting training examples
![image_20240218_181700.png](image_20240218_181700.png)

```tex
y=\sum_{c=1}^{C}\sum_{k_{1}=1}^{K}\sum_{k_{2}=1}^{K}\widehat{\mathcal{W}}_{c,k_{1},k_{2}}\times x_{c,k_{1},k_{2}}+b.
```

Now, if we further define:
```tex
\hat{x}_c=\sum_{k_1=1}^K\sum_{k_2=1}^K\widehat{\mathcal{W}}_{c,k_1,k_2}\times x_{c,k_1,k_2},\\
```

```tex
\hat{y}=\sum_{c=1}^C\hat{x}_c=y-b.
```

It is worthwhile to keep in mind that xˆ
and yˆ are random variables whose instantiations require fixed
spatial locations indexed by c, k1 and k2.

Given an input image, we first apply the CNN model in
the forward run to **find the input and output** of layer i + 1.
Then for any feasible (c, k1, k2) triplet,
we can obtain a C-dimensional vector variable ˆx = {xˆ1, xˆ2, . . . , xˆC } and a
scalar value yˆ using Equations mentioned above.

#### A greedy algorithm for channel selection

Objective Function:

```tex
\begin{aligned}\arg\min&\sum_{i=1}^m\left(\sum_{j\in T}\mathbf{\hat{x}}_{i,j}\right)^2\\\text{s.t.}&\quad|T|=C\times(1-r),\quad T\subset\{1,2,\ldots,C\}.\end{aligned}
```

![image_20240218_184900.png](image_20240218_184900.png)

#### Minimize the reconstruction error

Now we will further minimize the reconstruction error
 by weighing the channels, which can be defined
as:

```tex
\mathbf{\hat{w}}=\arg\min_{\mathbf{w}}\sum_{i=1}^{m}(\hat{y}_{i}-\mathbf{w}^{\mathrm{T}}\mathbf{\hat{x}}_{i}^{*})^{2},
```

which is a classic linear regression problem

####  pruning strategies

> Different strategies adopted here corresponding to network architectures.

1. **Pruning for VGG-16**:
    - Identify that more than 90% of FLOPs exist in the first 10 layers (conv1-1 to conv4-3).
    - Notice that FC layers contribute nearly 86.41% parameters.
    - Prune the first 10 layers for acceleration, but replace the FC layers with a global average pooling layer.
    - Belief that removing FC layers is simpler and more efficient.

2. **Pruning for ResNet**:
    - Face restrictions due to the special structure of ResNet.
    - Each block in the same group must have consistent channel numbers to finish the sum operation.
    - Pruning the last convolutional layer of each residual block directly is difficult.
    - Since most parameters are located in the first two layers, pruning the first two layers is a good choice.

3. **Different Filter Selection Criteria**:
    - **Weight Sum**:
        - Filters with smaller kernel weights tend to produce weaker activations.
        - Calculate the absolute sum of each filter as its importance score.
    - **APoZ (Average Percentage of Zeros)**:
        - Calculate the sparsity of each channel in output activations as its importance score.
        - Measure the ratio of zeros to non-zero activations.

These strategies highlight different approaches to pruning convolutional neural networks, considering factors such as layer importance, parameter distribution, and activation characteristics.











