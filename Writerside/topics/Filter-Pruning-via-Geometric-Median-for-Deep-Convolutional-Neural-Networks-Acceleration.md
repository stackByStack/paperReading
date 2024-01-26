# Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration
<show-structure for="chapter,procedure" depth="3"/>

## Abstract
Previous works utilized **smaller-norm-less-important** criterion to prune filters with smaller norm values in a convolutional neural network.

Its effectiveness depends on two requirements that are not always met: (1) the **norm deviation** of the filters should be **large**; (2) the **minimum norm** of the filters should be **small**.

![image_20240122_220301.png](image_20240122_220301.png)

To solve this problem, we propose a novel filter pruning method, namely **Filter Pruning via Geometric Median (FPGM)**.

FPGM compresses CNN models by pruning filters with **redundancy**.

![image_20240122_220300.png](image_20240122_220300.png)

## Introduction

FPGM chooses the filters with **the most replaceable contribution**.

![image_20240123_131100.png](image_20240123_131100.png){thumbnail="true"}

## Related Works

### Matrix Decomposition

<a href="https://arxiv.org/abs/1505.06798">Accelerating very deep convolutional networks for classification and detection</a>

<a href="https://arxiv.org/abs/1511.06067">Convolutional neural networks with low-rank regularization</a>

### low-precision weights
- Trained ternary quantization
- Incremental network quantization: Towards lossless CNNs with low-precision weights
  - On one hand, we introduce three interdependent operations, namely **weight partition, group-wise quantization and re-training**. 
  - A well-proven measure is employed to divide the weights in each layer of a pre-trained CNN model into **two disjoint groups.** 
  - The weights in the first group are responsible to **form a low-precision base**, thus they are quantized by a **variable-length encoding method**. 
  - The weights in the other group are responsible to **compensate** for the accuracy loss from the quantization, thus they are the ones to be re-trained. 
  - On the other hand, these three operations are **repeated** on the latest re-trained group in an iterative manner until **all the weights** are converted into low-precision ones, acting as an **incremental** network quantization and accuracy enhancement procedure. 

- Clustering convolutional kernels to compress deep neural networks
  - we perform clustering on
    3 × 3 kernels and **replace the redundant kernels with their centroids**. 
  - Therefore,
    we represent the compressed model with a set of centroids and a corresponding
    cluster index per each kernel. 
  - Thus, kernels that have the same index **share their
    weights.** While maintaining the compressed state, we train our model through
    the weight-sharing.
  - Our compression method brings following contributions and benefits. 
    - First,
      we propose a new method to compress and accelerate the CNN by **applying k-means clustering to 2D kernels**. 
    - Second, our **transform invariant clustering** method extends the valid
      number of kernel centroids with geometric transforms.

### knowledge distilling

### pruning

- weight pruning (often causing unstructured problems)
- Data Dependent Filter Pruning
- Data Independent Filter Pruning

## Method

### Preliminaries

In this setting, we want to find the filter

![image_20240123_1312.png](image_20240123_1312.png)

### Filter Pruning via Geometric Median

Thus, the algorithm is as follows.

![image_20240123_131300.png](image_20240123_131300.png)





