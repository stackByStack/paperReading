# EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning

## Abstract
<tldr>
In this
work, we present a pruning method called EagleEye, in which a simple
yet an efficient evaluation component based on adaptive batch normalization is applied to unveil a strong correlation between different pruned
DNN structures and their final settled accuracy.

</tldr>

## Introduction
![image_20240210_121900.png](image_20240210_121900.png)

- However, we found that the evaluation methods in existing works are suboptimal. Concretely, they are either inaccurate or complicated.

<a href="https://arxiv.org/abs/1802.03494">AutoML for Model Compression and Acceleration on Mobile Devices via reinforcement learning</a>

- To our knowledge, we are the
first to introduce correlation-based analysis for sub-net selection in pruning task.
- Moreover, we demonstrate that the reason such evaluation is inaccurate is the
  use of **suboptimal statistical values for Batch Normalization (BN) layers**.
- In this work, we use a so-called adaptive BN technique to fix the issue and
  effectively reach a **higher correlation** for our proposed evaluation process.

<a href="https://arxiv.org/abs/1804.03230">NetAdapt - greedy method based</a>

![image_20240210_133100.png](image_20240210_133100.png)

<a href="https://arxiv.org/abs/1608.03665">Introduces group-LASSO to introduce sparsity of
the kernels</a>

<a href="https://arxiv.org/abs/1708.06519">directly leverage the γ parameters in
BN layers as the scaling factors</a>

<a href="https://arxiv.org/abs/1903.09291">Towards Optimal Structured CNN Pruning via Generative Adversarial Learning</a>

## Methodology
![image_20240210_151100.png](image_20240210_151100.png)

Different searching methods have been applied in previous work to find the optimal pruning strategy, such as **greedy algorithm** [26,28], **RL** [7], and **evolutionary algorithm** [20]. All of these methods
are guided by the evaluation results of the pruning strategies.

### Motivation 
In many published approaches [7,13,19] in this domain, pruning candidates directly compare with each other in terms of **evaluation accuracy**. The subnets
with higher evaluation accuracy are selected and **expected to also deliver high accuracy after fine-tuning.** However, such intention **cannot** be necessarily achieved
as we notice the subnets perform poorly if directly used to do inference.

- why removal to filters, especially considered as unimportant filters, can cause
  such noticeable accuracy degradation, although the pruning rates are random?
- how strongly the low-range accuracy is positively correlated to the final converged accuracy

![image_20240210_160200.png](image_20240210_160200.png)

Figure 3 right shows that it might not be the weights that mess up
the **accuracy** at the evaluation stage as **only a gentle shift** in weight distribution is observed during fine-tuning, but the delivered inference **accuracy** is **very
different**.

![image_20240210_160600.png](image_20240210_160600.png)

So the layer-wise feature map
data are also affected by the changed model dimensions. However, vanilla evaluation **still uses Batch Normalization (BN) inherited from the full-size model**.
The **outdated** statistical values of BN layers eventually **drag down** the evaluation accuracy to a surprisingly low range. And, more importantly, break the
correlation between evaluation accuracy and the final converged accuracy of the
pruning candidates in the strategy-searching space.

#### BN Basics
![image_20240210_162800.png](image_20240210_162800.png)

### Adaptive Batch Normalization

If
the global BN statistics are out-dated to the subnets, we should re-calculate µ_T
and σ^2_T with adaptive values by conducting a few iterations of inference on part of
the **training set**, which essentially **adapts the BN** statistical values to the pruned
network connections. Concretely, we **freeze all the network parameters** while
**resetting the moving average statistics**.

### EagleEye pruning algorithm
![image_20240210_180700.png](image_20240210_180700.png)

#### Strategy generation
Concretely, it randomly samples
L real numbers from a given range [0, R] to form a pruning strategy, where r_l denotes the pruning ratio for the l
th layer. 






