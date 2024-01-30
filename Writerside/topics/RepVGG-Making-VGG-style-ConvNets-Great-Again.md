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