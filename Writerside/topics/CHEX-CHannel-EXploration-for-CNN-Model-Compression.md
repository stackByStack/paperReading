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

- Efficient image super
  resolution via channel discriminative deep neural network
  pruning

![image_20240213_104000.png](image_20240213_104000.png)

- Non-Structured DNN Weight Pruning—Is It Beneficial in Any Platform?
  - non-structured pruning is not competitive in terms of storage and computation efficiency

- <a href="https://arxiv.org/abs/1804.03230">NetAdapt - greedy method based</a>

- Good Subnetworks Provably Exist: Pruning via Greedy Forward Selection

![image_20240213_163200.png](image_20240213_163200.png)

![image_20240213_172400.png](image_20240213_172400.png)

- Gate decorator: Global filter pruning method for accelerating deep convolutional neural networks

![image_20240214_102100.png](image_20240214_102100.png)

![image_20240214_102400.png](image_20240214_102400.png)

![image_20240214_102500.png](image_20240214_102500.png)

- StructADMM: Achieving Ultrahigh Efficiency in
  Structured Pruning for DNNs
  -  The
     proposed framework incorporates stochastic gradient descent
     (SGD; or ADAM) with alternating direction method of multipliers (ADMM) and can be understood as a dynamic regularization
     method in which the regularization target is analytically updated
     in each iteration

- Discrimination-aware Channel Pruning
  for Deep Neural Networks

![image_20240214_164200.png](image_20240214_164200.png)

![image_20240214_164000.png](image_20240214_164000.png)

### Training-based pruning
- Neuron-level Structured Pruning using Polarization
  Regularizer
  - We propose a novel regularizer, namely polarization, for structured pruning of neural
    networks.
  - We theoretically analyzed the properties of polarization regularizer and proved
    that it simultaneously pushes a proportion of scaling factors to 0 and others to values larger
    than 0.
  - ![image_20240214_174500.png](image_20240214_174500.png)
- <a href="https://arxiv.org/abs/1708.06519">directly leverage the γ parameters in BN layers as the scaling factors</a>
- <a href="https://arxiv.org/abs/1903.09291">Towards Optimal Structured CNN Pruning via Generative Adversarial Learning</a>
- <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Compressing_Convolutional_Neural_Networks_via_Factorized_Convolutional_Filters_CVPR_2019_paper.pdf">Compressing Convolutional Neural Networks via Factorized Convolutional Filters</a>
  -  In this work, we propose to conduct **filter selection and filter learning** simultaneously, in a unified
     model. 
  - To this end, we define a **factorized convolutional
     filter (FCF)**,
    consisting of **a standard real-valued convolutional filter and a binary scalar**,
    as well as a **dot-product operator** between them.
    
  - We train a CNN model with factorized
     convolutional filters (CNN-FCF) by updating the standard
     filter using back-propagation, while updating the _binary
     scalar_ using the alternating direction method of multipliers (**ADMM**) based optimization method.
  - With this trained
     CNN-FCF model, we **only keep the standard filters corresponding to the 1-valued** scalars,
    while all other filters and
     all binary scalars are discarded, to obtain a compact CNN
     model. 
- <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Multi-Dimensional_Pruning_A_Unified_Framework_for_Model_Compression_CVPR_2020_paper.pdf">Multi-Dimensional Pruning: A Unified Framework for Model Compression</a>
  - Our MDP framework consists of three stages: the searching stage, the pruning stage, and the fine-tuning stage. 
  - In the searching stage, we firstly construct an over-parameterized network from any given original network to
  be pruned and then train this over-parameterized network
  by using the objective function introduced in Sec.
  3.2.2.
  - In the pruning stage, we prune the unimportant branches
  and channels in this over-parameterized network **based on
  the importance scores learned in the searching stage.**
  - In the
  fine-tuning stage, we fine-tune the pruned network to recover from the accuracy drop.

```tex
\begin{aligned}\mathrm{arg}\min_{\mathbf{\Theta},\mathbf{\lambda},\mathbf{G}}\mathcal{L}&=\mathcal{L}_{c}+\alpha\mathcal{L}_{st}+\eta\mathcal{L}_{gate},\\\\\mathrm{where~}\mathcal{L}_{st}&=\sum_{l=1}^{L}\sum_{i=1}^{B^{(l)}}\mathcal{S}(\lambda_{i}^{(l)})\cdot\mathcal{F}(\mathcal{T}_{i}^{(l)}),\\\mathcal{L}_{gate}&=\sum_{l=1}^{L}\sum_{i=1}^{B^{(l)}}\sum_{k=1}^{c_{out}}\|g_{i,k}^{(l)}\|_{1}.\end{aligned}
```

![image_20240215_102100.png](image_20240215_102100.png){thumbnail="true"}

> However, these commonly adopted regularization
may not penalize the parameters to exactly zero.
> Removing many _small but non-zero_ parameters inevitably **damages** the
model accuracy

- Specialized Optimizer
- <a href="https://arxiv.org/pdf/1904.03837.pdf">Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure</a>
  - However, instead of zeroing out filters, which ends up with a pattern where some whole filters are close to zero, we intend to **merge multiple
filters into one**, leading to a redundancy pattern where some
filters are identical.

### Difference
In contrast to the traditional pruning approaches that permanently remove channels, 
we dynamically adjust the importance
of the channels via a **_periodic pruning and regrowing process_**, which allows 
the prematurely pruned channels to be
recovered and prevents the model from losing the representation ability early in the training process.

![image_20240215_185800.png](image_20240215_185800.png){thumbnail="true"}

## Methodology

![image_20240215_193200.png](image_20240215_193200.png)

![image_20240216_002010.png](image_20240216_002010.png)

```tex
\epsilon_{j}^{l}=\|\mathbf{w}_{j}^{l}-\mathbf{w}_{\mathcal{T}^{l}}^{l}(\mathbf{w}_{\mathcal{T}^{l}}^{l^{T}}\mathbf{w}_{\mathcal{T}^{l}}^{l})^{\dagger}\mathbf{w}_{\mathcal{T}^{l}}^{l^{T}}\mathbf{w}_{j}^{l}\|_{2}^{2}
```















