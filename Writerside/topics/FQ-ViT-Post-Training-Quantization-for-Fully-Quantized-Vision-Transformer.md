# FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer

<show-structure for="chapter,procedure" depth="3"/>

## Abstract

- most existing
  quantization methods have been developed mainly
  on Convolutional Neural Networks (CNNs), and
  suffer severe degradation when applied to **fully
  quantized** vision transformers.
- In this work, we
  demonstrate that many of these difficulties _arise
  because of_ serious **inter-channel variation** in LayerNorm inputs, and present, Power-of-Two Factor (PTF), a systematic method to **reduce the performance degradation and inference complexity** of
  fully quantized vision transformers.
- In addition,
  observing an _extreme_ **non-uniform distribution** in
  attention maps, we propose **Log-Int-Softmax (LIS)**
  to **sustain** that and simplify inference by using **4-bit quantization** and the **BitShift operator**.

## Introduction

### Up or Down? Adaptive Rounding for Post-Training Quantization
<a href="https://arxiv.org/abs/2004.10568"></a>

- we propose AdaRound, a better weight-rounding mechanism for post-training quantization that adapts to the data and the task loss.
- AdaRound is fast, does not require fine-tuning of the network, and only uses a small amount of unlabelled data.
- By approximating the **task loss** with a **Taylor series expansion**, the rounding task is posed as a **quadratic unconstrained binary optimization problem.**
- We simplify this to a layer-wise local loss and propose to optimize this loss with a soft relaxation. 

Rounding-to-nearest is the predominant approach for all
neural network weight quantization work that came out thus
far.

```tex
\widehat{\mathbf{w}}=\mathrm{s}\cdot clip\left(\left\lfloor\frac{\mathbf{w}}{\mathbf{s}}\right\rceil,\mathrm{n},\mathrm{p}\right)
```

####  Task-loss-based rounding

To **avoid** the computational overhead of **repeated forward**
passes through the data, we utilize **the second order Taylor
series approximation**. Additionally, we **ignore** the interactions among weights belonging to **different layers**. 

This, in turn, implies that we assume a block **diagonal** H(w)
, where
each non-zero block corresponds to one layer. We thus end
up with the following per-layer optimization problem.

```tex
\underset{\Delta\mathbf{w}^{(\ell)}}{\text{arg}\operatorname*{\min}}\quad\mathbb{E}\left[\mathbf{g}^{(\mathbf{w}^{(\ell)})^T}\Delta\mathbf{w}^{(\ell)}+\frac12\Delta\mathbf{w}^{(\ell)^T}\mathbf{H}^{(\mathbf{w}^{(\ell)})}\Delta\mathbf{w}^{(\ell)}\right].
```


For a converged pretrained model, the contribution of the gradient term for
optimization can be safely ignored. This results in
```tex
\underset{\Delta \mathbf{w}^{(\ell)}}{\arg \min } \mathbb{E}\left[\Delta \mathbf{w}^{(\ell)^{T}} \mathbf{H}^{\left(\mathbf{w}^{(\ell)}\right)} \Delta \mathbf{w}^{(\ell)}\right]
```

#### From Taylor expansion to local loss
> local means decision in a layer-wise form and based on information only from a single layer
> 
> This is for avoiding high computation complexity.
> 

#### AdaRound

We take this as objective:
```tex
\underset{\mathbf{V}}{\text{arg}\operatorname*{min}}\quad\left\|\mathbf{W}\mathbf{x}-\widetilde{\mathbf{W}}\mathbf{x}\right\|_{F}^{2}+\lambda f_{reg}\left(\mathbf{V}\right)
```

where ||·||^2 _F
denotes the Frobenius norm and W~ are the
soft-quantized weights that we optimize over

```tex
\widetilde{\mathbf{W}}=\mathrm{s} \cdot \operatorname{clip}\left(\left\lfloor\frac{\mathbf{W}}{\mathrm{s}}\right\rfloor+h(\mathbf{V}), \mathrm{n}, \mathrm{p}\right)
```

V_i,j is the continuous
variable that we optimize over and h (Vi, j) can be **any differentiable function** that takes values between 0 and 1, i.e.,
h (V_i, j) ∈ [0, 1]. 

The additional term f_reg (V) is a differentiable regularizer that is introduced to encourage the
optimization variables h (Vi, j) to converge towards either
0 or 1, i.e., at convergence h (Vi,j) ∈ {0, 1}.

The rectified sigmoid is defined as
```tex
h\left(\mathbf{V}_{i, j}\right)=\operatorname{clip}\left(\sigma\left(\mathbf{V}_{i, j}\right)(\zeta-\gamma)+\gamma, 0,1\right)

```
where σ(·) is the sigmoid function and, ζ and γ are stretch
parameters, fixed to 1.1 and −0.1, respectively.

Then we would perform the quantization layer by layer.

### Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT
<a href="https://arxiv.org/pdf/1909.05840.pdf"></a>

In
this work, we perform an extensive analysis of fine-tuned BERT models using **second order Hessian information**, and we use our results to propose
a novel method for quantizing BERT models to ultra low precision. In particular, we propose a new **group-wise quantization scheme**, and we use a
Hessian-based **mix-precision** method to compress the model further.

A <a href="https://arxiv.org/abs/1905.03696">Hessian AWare Quantization (HAWQ)</a> is developed for mixed-bits assignments. The main idea is that the
parameters in NN layers **with higher Hessian spectrum** (i.e., _larger top eigenvalues_) are **_more sensitive_** to quantization and
**require higher precision**, as compared to layers with small Hessian spectrum (i.e., smaller top eigenvalues).

#### Group-wise Quantization

![image_20240307_171700.png](image_20240307_171700.png){thumbnail="true"}

### Revisit LayerNorm and Softmax

Firstly, we find a serious inter-channel variation of LayerNorm inputs, which some channel
**ranges** even exceed **40×** of the median. Traditional methods
**cannot handle** such **large fluctuations of _activations_**, which
will lead to large quantization error.

Secondly, we find that
the values of the attention map have an **_extreme_** **non**-_uniform_
distribution, with **most** values clustered in 0 ∼ 0.01, and a few
high attention values close to 1.

## Methodology
### Preliminary
Assuming the quantization bit-width is b, the quantizer
Q(X|b) can be formulated as a function that maps a floating-point number X ∈ R to the nearest quantization bin:

```tex
\mathrm{Q}(\mathrm{X}|b):\mathbb{R}\to\mathrm{q},
```

```tex
\mathrm{q}=\left\{\begin{array}{lr}\left\{-2^{b-1}, \cdots, 2^{b-1}-1\right\} & \text { Signed } \\ \left\{0,1 \cdots, 2^{b}-1\right\} & \text { Unsigned }\end{array}\right.
```

#### Quantization Styles
**Uniform Quantization** is well-supported on most hardware platforms. Its quantizer Q(X|b) can be defined as:
```tex
\mathrm{Q}(\mathrm{X}|b)=\mathrm{clip}(\lfloor\frac{\mathrm{X}}{s}\rceil+zp,0,2^b-1)
```

where s (scale) and zp (zero-point) are quantization parameters determined by the lower bound l and the upper bound u
of X, which are usually minimum and maximum:

```tex
\begin{aligned}
&l=\min (\mathrm{X}), u=\max (\mathrm{X})\\
&s=\frac{u-l}{2^{b}-1}, z p=\operatorname{clip}\left(\left\lfloor-\frac{l}{s}\right\rceil, 0,2^{b}-1\right)
\end{aligned}
```

**Log2 Quantization** converts the quantization process from
linear to exponential variation. Its quantizer Q(X|b) can be
defined as:
```tex
\mathrm{Q}(\mathrm{X}|b)=\mathrm{sign}(\mathrm{X})\cdot\mathrm{clip}(\left\lfloor-\log_2\frac{|\mathrm{X}|}{\max(|\mathrm{X}|)}\right\rceil,0,2^{b-1}-1).
```

###  Power-of-Two Factor for LayerNorm Quantization

During inference, LayerNorm [Ba et al., 2016] computes the
statistics µX, σX **in each forward step** and **normalizes input** X.

The above process can be written as:
```tex
\text{LayerNorm(X)}=\frac{\mathrm{X}-\mu_\mathrm{X}}{\sqrt{\sigma_\mathrm{X}^2+\epsilon}}\cdot\gamma+\beta.
```

It is observed that the channel-wise ranges fluctuate more
wildly in vision transformers than those in ResNets.

Based on such extreme inter-channel variation, layer-wise
quantization, which applies **the same quantization parameters**
to **all channels**, will lead to an intolerable quantization error.

A possible solution is using group-wise quantization or  channel-wise quantization.

The core idea of Power-of-Two Factor(PTF) is to equip different channels with
different factors, rather than different quantization parameters.

Given the quantization bit-width b, the input activation X ∈ R^B×L×C
, the layer-wise quantization parameters
s, zp ∈ R^1
, and the PTF α ∈ N^C
, then the quantized activation X_Q can be formulated as:
```tex
\mathrm{X_Q}=\mathrm{Q}(\mathrm{X}|b)=\mathrm{clip}(\lfloor\frac{\mathrm{X}}{2^\alpha s}\rceil+zp,0,2^b-1)
```

with 
```tex
\begin{aligned}
&s=\frac{\max(\mathbf{X})-\min(\mathbf{X})}{2^b-1}/2^\mathbf{K},\\
&zp=\operatorname{clip}(\lfloor-\frac{\min(\mathbf{X})}{2^\mathbf{K}s}\rceil,0,2^b-1),\\
&\alpha_c=\underset{\alpha_c\in\{0,1,\cdots,\mathbf{K}\}}{\operatorname*{\arg\min}}\left\|\mathrm{X}_c-\lfloor\frac{\mathrm{X}_c}{2^{\alpha_c}s}\rceil\cdot2^{\alpha_c}s\right\|_2.
\end{aligned}
```

Noticing c represents the **channel index** for X and α. The
hyperparameter K could meet **different scaling requirements**.

To **_cover_** the different inter-channel variation across
all models, we set K = 3 as default. 

#### BitShift
Meanwhile, thanks to the nature of
powers of two, PTF α can be efficiently combined with 
layer-wise quantization by BitShift operator, 
avoiding floating-point calculations of group-wise or channel-wise quantization. 
The whole process can be processed with two phases:

**Phase 1:**

```tex
\widehat{\mathrm{X}}_{\mathrm{Q}}=(\mathrm{X}_{\mathrm{Q}}-zp)<<\alpha.
```
**Phase 2:**

```tex
\begin{aligned}\mu(\mathrm{X})&\approx\mu(2^\alpha s\cdot(\mathrm{X}_\mathrm{Q}-zp))=s\cdot\mu(\widehat{\mathrm{X}}_\mathrm{Q}),\\\sigma(\mathrm{X})&\approx\sigma(2^\alpha s\cdot(\mathrm{X}_\mathrm{Q}-zp))=s\cdot\sigma(\widehat{\mathrm{X}}_\mathrm{Q}).\end{aligned}
```

### Log-Int-Softmax for Softmax Quantization
#### Log2 Quantization for Attention Map

We quantize attention maps to lower bit-width.

Inspired by the idea of sparse attention in DynamicViT [Rao et al., 2021], we probe into the **distribution of
attention maps**.

![image._20240308_001200.png](image._20240308_001200.png)








