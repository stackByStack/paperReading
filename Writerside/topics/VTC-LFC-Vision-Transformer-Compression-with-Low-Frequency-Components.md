# VTC-LFC: Vision Transformer Compression with Low-Frequency Components

## Abstract
The compression only
in the spatial domain suffers from a dramatic performance drop without fine-tuning and is **not robust** to noise.

Because the noise in the spatial domain can easily
**confuse the pruning criteria**, leading to some parameters/channels being pruned
incorrectly.

Inspired by recent findings that self-attention is a **low-pass filter** and
low-frequency signals/components are more informative to ViTs, this paper proposes compressing ViTs with **low-frequency components**.

Two metrics named
**low-frequency sensitivity** (LFS) and **low-frequency energy** (LFE) are proposed
for better channel pruning and token pruning.

## Introduction
Token pruning based on low-frequency energy:
Token compression/sampling aims to select the informative tokens that store **more useful information**.

The _popular methods_ dynamically select those tokens with high correlation to other tokens (e.g. the
CLS token) as the informative tokens.
However, it may be sub-optimal because the selected tokens
tend to be similar to each other, and the information included in the token itself has been neglected to
some extent.

## Methodology

Our goal is to
reduce the channel number of linear projection matrices and the token number N^l.

![image_20240301_090500.png](image_20240301_090500.png)

### Channel Pruning based on Low-Frequency Sensitivity

The importance score I_j of a weight w_j is formulated as:

```tex
\mathcal{I}_j=\left(\mathcal{L}\left(\mathcal{M}\left(X,\mathbf{W}\right),Y\mid w_j=0\right)-\mathcal{L}\left(\mathcal{M}\left(X,\mathbf{W}\right),Y\right)\right)^2
```

(cross-entropy loss in this
paper)

The low-frequency components in images
X˜ are formulated as

```tex
\tilde{X}=\mathcal{F}^{-1}\left(\mathcal{G}\left(\sigma_c\right)\odot\mathcal{F}\left(X\right)\right)
```

G (·) is the low-pass filter. Considering that a binary filter will cause the Ringing
effect when the image is transformed back to the spatial domain, Gaussian filter is chosen for G (·).

In addition to the task-specific loss, the pruned model shall also provide robust feature representation
as the original model. In other words, the feature representation of the low-frequency images
X˜ shall be as close to that of the original images X as possible.

Hence, apart from the cross-entropy loss L for the classification task, a knowledge-distillation loss is also taken into account

We can measure the error between the CLS tokens
corresponding to the low-frequency image and nature image, respectively.

Then, Low-Frequency Sensitivity (LFS) is formulated as:
```tex
s_j=\lambda\cdot\left(\mathcal{L}(\tilde{X}\mid w_j=0)-\mathcal{L}(\tilde{X})\right)^2+(1-\lambda)\cdot\left(\mathcal{K}\mathcal{L}(\tilde{T},T\mid w_j=0)-\mathcal{K}\mathcal{L}(\tilde{T},T)\right)^2
```

where λ is the hyper-parameter for the balance of two loss functions.

Calculating the LFS for each parameter is **infeasible** for models _with millions of
parameters_.

```tex
\hat s_j=\lambda\cdot\left(\frac{\partial\mathcal{L}(\tilde{X})}{\partial w_j}\cdot w_j\right)^2+(1-\lambda)\cdot\left(\frac{\partial\mathcal{K}\mathcal{L}(\tilde{T},T)}{\partial w_j}\cdot w_j\right)^2,
```

The LFS of a channel is computed by the _**sum**_ of sˆj.

### Token Pruning based on Low-Frequency Energy

we evaluate
the low-frequency ratio of the token after transforming tokens X^l,2
into the frequency domain by
applying FFT on each channel of tokens, denoted as

```tex
\mathcal{X}_{b,:,j}^{l,2}=\mathcal{F}\left(X_{b,:,j}^{l,2}\right)
```

Given filter G with a cutoff factor σ_t, the LFE is formulated as:

```tex
\eta_{l,i}=\frac{\left\|\mathcal{LC}\left[\mathcal{X}^{l,2}\right]\right\|_{2}}{\left\|\mathcal{DC}\left[\mathcal{X}^{l,2}\right]\right\|_{2}}=\frac{\left\|\mathcal{F}^{-1}(\mathcal{G}\left(\sigma_{t}\right)\odot\mathcal{X}_{b,i,\cdot}^{l,2})\right\|_{2}}{\left\|\mathcal{F}^{-1}\left(\mathcal{X}_{b,\cdot,\cdot}^{l,2}\right)\right\|_{2}}=\frac{\left\|\tilde{X}_{b,i,\cdot}^{l,2}\right\|_{2}}{\left\|X_{b,\cdot,\cdot}^{l,2}\right\|_{2}}

```

For the h-th head in ViT, the attention value is calculated as:

```tex
\mathcal{A}^{l,h}=softmax\left(\frac{Q_{l,h}K_{l,h}^T}{\sqrt{d_{l,h}}}\right)
```

The CLS token plays a
more significant role than other tokens because it is the _final output feature_ that collects information.

```tex
\hat{\mathcal{T}}_{l,i}=\frac{1}{H}\sum_{h=0}^{H-1}\left(\theta_{h,0}\cdot\mathcal{A}_{i,0}^{l,h}+\theta_{h,1}\cdot\frac{1}{N^{l}}\sum_{j=1}^{N^{l}-1}\mathcal{A}_{i,j}^{l,h}\right)

```

![image_20240301_104800.png](image_20240301_104800.png)

To estimate the importance score of tokens from multiple and diverse aspects, we consider to combine
the LFE η_l,i and attention score Tˆ
l,i to get the final importance score of a token as:

```tex
\tilde{\mathcal{T}}_{l,i}=\hat{\mathcal{T}}_{l,i}\cdot\eta_{l,i}
```

### Bottom-up Cascade Pruning

![image_20240301_105600.png](image_20240301_105600.png)







