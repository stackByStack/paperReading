# DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification

## Abstract
- Specifically, we devise a lightweight prediction
  module to estimate the importance score of each token given the current features.
  The module is added to different layers to prune redundant tokens hierarchically. 
- To
  optimize the prediction module in an end-to-end manner, we propose an attention
  masking strategy to differentiably prune a token by blocking its interactions with
  other tokens.

## Introduction
- Considering the number of zero elements
in the binary decision mask is different for each instance, **_directly eliminating_** the uninformative
tokens for each input instance during training will make parallel computing impossible. 
- Moreover,
this would also **hinder** **the back-propagation** for the prediction module, which needs to calculate the
probability distribution of whether to keep the token even if it is finally eliminated. 
- Besides, directly
setting the abandoned tokens as zero vectors is also not a wise idea since zero vectors will **still affect**
the calculation of the attention matrix. Therefore, we propose a strategy called **_attention masking_**
where we drop the connection from abandoned tokens to all other tokens in the attention matrix based
on the binary decision mask.

##  Dynamic Vision Transformers

### Overview

![image_20240301_165100.png](image_20240301_165100.png)

### Hierarchical Token Sparsification with Prediction Modules

We initialize all elements in the decision
mask to 1 and update the mask progressively. The prediction modules take the current **decision** Dˆ
and the tokens x ∈ R ^N×C as input. We first project the tokens using an MLP:

```tex
\mathbf{z}^{\mathrm{local}}=\mathrm{MLP}(\mathbf{x})\in\mathbb{R}^{N\times C^{\prime}}
```

where C'
can be a smaller dimension, and we use C' = C/2 in our implementation. Similarly, we
can compute a global feature by:

```tex
\mathbf{z}^{\mathrm{global}}=\mathrm{Agg}(\mathrm{MLP}(\mathbf{x}),\hat{\mathbf{D}})\in\mathbb{R}^{C^{\prime}}
```

where Agg is the function which aggregates the information all the existing tokens and can be simply
implemented as an average pooling:

```tex
\mathrm{Agg}(\mathbf{u},\hat{\mathbf{D}})=\frac{\sum_{i=1}^N\hat{\mathbf{D}}_i\mathbf{u}_i}{\sum_{i=1}^N\hat{\mathbf{D}}_i},\quad\mathbf{u}\in\mathbb{R}^{N\times C^{\prime}}
```

```tex
\begin{aligned}
\mathbf{z}_i&=[\mathbf{z}_i^\mathrm{local},\mathbf{z}_i^\mathrm{global}],\quad1\leq i\leq N,\\
 \pi&=\mathrm{Softmax}(\mathrm{MLP}(\mathbf{z}))\in\mathbb{R}^{N\times2}
 \end{aligned}
```

where π_i,0 denotes the probability of **dropping** the i-th token and π_i,1 is the probability of **keeping** it.

We can then generate current decision D by **sampling** from π and update Dˆ by

```tex
\hat{\mathrm{D}}\leftarrow\hat{\mathrm{D}}\odot\mathrm{D}
```

indicating that once a token is dropped, it will never be used.

>We omit the class token for simplicity, while in practice we always keep the class token (i.e., the decision
for class token is always “1”)
> 

### End-to-end Optimization with Attention Masking

First, the sampling from π to get binary decision mask D is non-differentiable,
which impedes the end-to-end training. 

To overcome this, we apply the **Gumbel-Softmax technique** to sample from the probabilities π

```tex
\mathbf{D}=\text{Gumbel-Softmax}({\pi})_{*,1}\in\{0,1\}^N,
```
The output of Gumbel-Softmax is a one-hot tensor.

To achieve parallelized computation, we must **keep** the
number of tokens unchanged to avoid simple yet unstructured pruning.

The zeroed tokens will **still influence** other tokens through the **Softmax operation**. To this end, we
devise a strategy called attention masking which can totally eliminate the effects of the dropped
tokens. Specifically, we compute the attention matrix by:

```tex
\begin{aligned}
&\mathbf{P}=\mathbf{Q}\mathbf{K}^T/\sqrt{C}\in\mathbb{R}^{N\times N}, \\
&\mathbf{G}_{ij}=\begin{cases}1,&i=j,\\\hat{\mathbf{D}}_j,&i\neq j.\end{cases}&& 1\leq i,j\leq N,  \\
&\tilde{\mathbf{A}}_{ij}=\frac{\exp(\mathbf{P}_{ij})\mathbf{G}_{ij}}{\sum_{k=1}^N\exp(\mathbf{P}_{ik})\mathbf{G}_{ik}},&& 1\leq i,j\leq N. 
\end{aligned}
```

### Training and Inference

To minimize the influence on performance caused by our token sparsification, we use the original
backbone network as a **_teacher model_** and hope the behavior of our DynamicViT as close to the
teacher model as possible. 

First, we make
the **finally remaining tokens of the DynamicViT** _close to_ the ones of the **teacher** model, which can be
viewed as a kind of _self-distillation_:

```tex
\mathcal{L}_{\mathrm{distill}}=\frac1{\sum_{b=1}^B\sum_{i=1}^N\hat{\mathbf{D}}_i^{b,S}}\sum_{b=1}^B\sum_{i=1}^N\hat{\mathbf{D}}_i^{b,S}(\mathbf{t}_i-\mathbf{t}_i^{\prime})^2
```

where t_i and t'_i
denote the i-th token after the last block of the DynamicViT and the teacher model,
respectively. Dˆ b,s is the decision mask for the b-th sample at the s-th sparsification stage.

Second,
we minimize the difference of the predictions between our DynamicViT and its teacher via the KL
divergence.

Given a set of target
ratios for S stages ρ = [ρ
(1), . . . , ρ(S)
], we utilize an MSE loss to supervise the prediction module:

```tex
\mathcal{L}_\text{ratio}=\frac{1}{BS}\sum_{b=1}^{B}\sum_{s=1}^{S}\left(\rho^{(s)}-\frac{1}{N}\sum_{i=1}^{N}\hat{\mathbf{D}}_i^{b,s}\right)^2.
```

Then we combine the losses mentioned above:
```tex
\mathcal{L}=\mathcal{L}_\mathrm{cls}+\lambda_\mathrm{KL}\mathcal{L}_\mathrm{KL}+\lambda_\mathrm{distill}\mathcal{L}_\mathrm{distill}+\lambda_\mathrm{ratio}\mathcal{L}_\mathrm{ratio}
```

**During inference**, given the target ratio ρ, we can directly discard the less informative tokens via the
probabilities produced by the prediction modules such that only exact m^s = ρ^s N tokens are kept
at the s-th stage. Formally, for the s-th stage, let

```tex
\mathcal{I}^s=\operatorname{argsort}({\pi}_{*,1})
```




