# RepVGG: Making VGG-style ConvNets Great Again

## Abstract
- a stack of 3 × 3 convolution and ReLU, while the training-time model has a **multi-branch topology**.
- **decoupling** of the training-time and inference-time architecture realized by **structural re-parameterization** technique

## Introduction

### Network Architecture Redesign
<a href="Comparative-Analysis-of-Classic-Convolutional-Neural-Networks-VGG-Inception-ResNet-and-DenseNet.md"></a>

### NAS
- Automatic
  - <a href="https://arxiv.org/abs/1707.07012"> learn model architectures directly on the dataset of interest</a>
  - <a href="https://arxiv.org/abs/1802.01548">regularized evolution to automatically discover image classifier architectures</a>
  - <a href="https://arxiv.org/abs/1712.00559">structures are searched in order of increasing complexity while simultaneously learning a surrogate model to guide the search through structure space.</a>

- Manually
  - <a href="https://arxiv.org/abs/2003.13678">developing progressively simplified versions of an initial design space</a>
  
### Searched Compound Scaling Strategy
<a href="https://arxiv.org/abs/1905.11946">balancing network depth, width, and resolution to improve performance</a>

### Drawback
- complicated multi-branch designs
  - Resnet: residual-addition
  - Inception: branch-concat
- Components increasing the memory access cost and lacking support of various devices.
  - depth-wise conv in Xception 
  - <a href="MobileNetv1-v3.md">MobileNets</a>
  - channel shuffle in ShuffleNets

### Advantage of RepVGG
RepVGG has the following advantages.
- The model has a VGG-like plain (a.k.a. feed-forward)
topology 1 without any branches, which means every
layer takes the output of its only preceding layer as
input and feeds the output into its only following layer. 
- The model’s body uses only 3 × 3 conv and ReLU. 
- The concrete architecture (including the specific depth
and layer widths) is instantiated with no automatic
search, manual refinement, compound scaling, nor other heavy designs.

### Motivation of decoupling between train-time and inference-time
Since **the benefits of multi-branch architecture are all
for training** and the drawbacks are undesired for inference, 
we propose to decouple the training-time multi-branch and inference-time plain architecture via **structural
re-parameterization**, which means converting the architecture from one to another via transforming its parameters.

![image_20240131_095600.png](image_20240131_095600.png)

## Building RepVGG via Structural Re-param

### Simple is Fast, Memory-economical and Flexible

- Many recent multi-branch architectures have
  lower theoretical FLOPs than VGG but may not run faster.
- The multi-branch topology imposes constraints on the architectural specification.

### Re-param for Plain Inference-time Model

![image_20240131_121900.png](image_20240131_121900.png)

![image_20240131_125100.png](image_20240131_125100.png)

This transformation also applies to the **identity branch**
because an identity can be viewed as a **1 × 1 conv** with an
identity matrix as the kernel.

After such transformations,
we will have one 3 × 3 kernel, two 1 × 1 kernels, and three
bias vectors.

Then we obtain the final bias by **adding up the
three bias vectors**, and the final 3 × 3 kernel by adding the
1×1 kernels onto the **central point of 3×3 kernel**, which can
be easily implemented by first zero-padding the two 1 × 1
kernels to 3 × 3 and adding the three kernels up.

###  Architectural Specification

We decide the numbers of layers of each stage following three simple guidelines.
- The first stage operates with
large resolution, which is time-consuming, so we use only
one layer for lower latency. 
- The last stage shall have
more channels, so we use only one layer to save the parameters. 
- We put the most layers into the second last stage

To further reduce the parameters and computations, we
may optionally **interleave** _groupwise 3 × 3 conv_ layers with
dense ones to trade accuracy for efficiency.

We **do
not** use **adjacent groupwise conv** layers because that would
**disable the inter-channel information exchange** and bring a
side effect: outputs from a certain channel would be
derived _from only a small fraction_ of input channels.














