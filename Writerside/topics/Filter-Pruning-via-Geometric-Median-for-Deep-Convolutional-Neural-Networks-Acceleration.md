# Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration

## Abstract
Previous works utilized **smaller-norm-less-important** criterion to prune filters with smaller norm values in a convolutional neural network.

Its effectiveness depends on two requirements that are not always met: (1) the **norm deviation** of the filters should be **large**; (2) the **minimum norm** of the filters should be **small**.

![image_20240122_220301.png](image_20240122_220301.png)

To solve this problem, we propose a novel filter pruning method, namely **Filter Pruning via Geometric Median (FPGM)**.

FPGM compresses CNN models by pruning filters with **redundancy**.

![image_20240122_220300.png](image_20240122_220300.png)

## Introduction

FPGM chooses the filters with **the most replaceable contribution**.


## Related Works

### Matrix Decomposition

<a href="https://arxiv.org/abs/1505.06798">Accelerating very deep convolutional networks for classification and detection</a>

<a href="https://arxiv.org/abs/1511.06067">Convolutional neural networks with low-rank regularization</a>

### low-precision weights
- Trained ternary quantization
- Incremental network quantization: Towards lossless cnns with lowprecision weights
  - On one hand, we introduce three interdependent operations, namely **weight partition, group-wise quantization and re-training**. 
  - A well-proven measure is employed to divide the weights in each layer of a pre-trained CNN model into **two disjoint groups.** 
  - The weights in the first group are responsible to **form a low-precision base**, thus they are quantized by a **variable-length encoding method**. 
  - The weights in the other group are responsible to **compensate** for the accuracy loss from the quantization, thus they are the ones to be re-trained. 
  - On the other hand, these three operations are **repeated** on the latest re-trained group in an iterative manner until **all the weights** are converted into low-precision ones, acting as an **incremental** network quantization and accuracy enhancement procedure. 




