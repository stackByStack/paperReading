# Post-Training Quantization for Vision Transformer

<show-structure for="chapter,procedure" depth="3"/>

## Abstract
-  In this paper, we present an effective **post-training quantization**
   algorithm for reducing the memory storage and computational costs of vision transformers.
- Basically, the quantization task can be regarded as finding the **optimal**
  _**low-bit**_ **quantization intervals** for weights and inputs, **respectively**.
- To preserve
  the functionality of the attention mechanism, we introduce a **ranking loss** into
  the conventional quantization objective that aims to **keep** the **relative order** of the
  **self-attention** results after quantization.
- Moreover, we thoroughly analyze the
  **relationship** between _quantization loss_ of **different layers** and the _feature_ **diversity**,
  and explore a mixed-precision quantization scheme by **exploiting the nuclear norm**
  of each attention map and output feature.

## Introduction
### Quantization in CNN
####  Trained ternary quantization
<a href="https://arxiv.org/pdf/1612.01064.pdf"></a>

In this paper, we propose Trained Ternary Quantization which uses two full-precision scaling
coefficients W
\^p _l
, W\^n
_l
for each layer l, and quantize the weights to {−W\^n
_l
, 0, +W\^
p
_l
} instead of
traditional {-1, 0, +1} or {-E, 0, +E}

#### HAQ: Hardware-Aware Automated Quantization with Mixed Precision
<a href="https://arxiv.org/pdf/1811.08886.pdf"></a>

![image_20240309_103000.png](image_20240309_103000.png)

#### PACT: Parameterized Clipping Activation for Quantized Neural Networks
<a href="https://arxiv.org/pdf/1805.06085.pdf"></a>

The modified activation is as follows.

```tex 
y=PACT(x)=0.5(|x|-|x-\alpha|+\alpha)=\begin{cases}0,&x\in(-\infty,0)\\x,&x\in[0,\alpha)\\\alpha,&x\in[\alpha,+\infty)\end{cases}
```

PACT: A new activation quantization scheme for finding the optimal quantization scale
during training. We introduce a new parameter α that is used to **represent the clipping level**
in the **activation function** and is learned via back-propagation. α sets the quantization scale
smaller than ReLU to reduce the quantization error, but larger than a conventional clipping
activation function (used in previous schemes) to allow gradients to flow more effectively.
In addition, **regularization is applied to α** in the loss function to enable faster convergence.


### Quantization in Transformer

#### Q-Bert
<a href="https://stackbystack.github.io/paperReading/fq-vit-post-training-quantization-for-fully-quantized-vision-transformer.html#q-bert-hessian-based-ultra-low-precision-quantization-of-bert">Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT</a>

#### TernaryBERT: Distillation-aware Ultra-low Bit BERT
<a href="https://arxiv.org/pdf/2009.12812.pdf">TernaryBERT: Distillation-aware Ultra-low Bit BERT</a>

![image_20240309_110000.png](image_20240309_110000.png){thumbnail="true"}

> However, these methods are **not** designed for computer vision tasks and usually need additional training or fine-tuning. 
> 
> Furthermore, in some scenarios, the **entire** training data is **not**
> available to optimize the quantization model and the training costs for edge devices are intolerable.
> 
> 


### Our Contributions
In this paper, we study the post-training quantization method for vision transformer models with
mixed-precision for higher compression and speed-up ratios. 

The quantized process in the transformer
is formulated as an optimization problem for finding the optimal quantization intervals. Specially,
our goal is to maximize the **similarity** between the **full-precision and quantized outputs** in vision
transformers. 

### Vision Transformer
#### Generative Pretraining from Pixels
<a href="https://dl.acm.org/doi/pdf/10.5555/3524938.3525096"></a>

![image_20240310_093000.png](image_20240310_093000.png)

#### Pre-trained image processing transformer
<a href="https://arxiv.org/pdf/2012.00364.pdf"></a>

![image_20240310_100300.png](image_20240310_100300.png){thumbnail="true"}

### Post-Training Quantization

#### ACIQ method
<a href="https://stackbystack.github.io/paperReading/resrep-lossless-cnn-pruning-via-decoupling-remembering-and-forgetting.html#analytical-clipping-for-integer-quantization-aciq"></a>

#### Towards Accurate Post-training Network Quantization via Bit-Split and Stitching

<a href="https://dl.acm.org/doi/pdf/10.5555/3524938.3525851"></a>

![image_20240310_104400.png](image_20240310_104400.png)

![image_20240310_104900.png](image_20240310_104900.png)

#### AdaRound
<a href="https://stackbystack.github.io/paperReading/fq-vit-post-training-quantization-for-fully-quantized-vision-transformer.html#adaround"></a>

#### Data-Free Quantization Through Weight Equalization and Bias Correction

<a href="https://arxiv.org/pdf/1906.04721.pdf"></a>

![image_20240310_110800.png](image_20240310_110800.png)

<a href="https://zhuanlan.zhihu.com/p/104052236"></a>

#### Zeroq: A novel zero shot quantization framework
<a href="https://arxiv.org/pdf/2001.00281.pdf"></a>

ZEROQ
enables **mixed-precision** quantization **without** any access to the
training or validation data.

This is achieved by optimizing for
a **Distilled Dataset**, which is engineered to match the statistics
of batch normalization across different layers of the network.

```tex
\min_{x^r}\sum_{i=0}^L\left\|\tilde{\mu}_i^r-\mu_i\right\|_2^2+\left\|\tilde{\sigma}_i^r-\sigma_i\right\|_2^2,
```

![image_20240310_162000.png](image_20240310_162000.png)

ZEROQ supports both **uniform** and **mixed-precision** quantization.
For the latter, we introduce a novel **Pareto frontier based** method
to automatically determine the mixed-precision bit setting for
all layers, with no manual search involved.

## Methodology

In this section, we elaborate on the proposed **mixed-precision** post-training quantization scheme
for the vision transformer.

The **similarity-aware** quantization for linear layers and **ranking-aware**
quantization for self-attention layers are presented.

In addition, the **bias correction** method for
optimization and the mixed-precision quantization based on **nuclear norm** of the _attention map_ and
_output feature_ are introduced.

![image_20240310_163800.png](image_20240310_163800.png){thumbnail="true"}

### Preliminaries

the input to the first transformer layer is

```tex
\begin{aligned}\mathbf{X}_1&=[x_{class};I_1^p\mathbf{W}_1^E;\cdots;I_n^p\mathbf{W}_n^E]+\mathbf{E}^{pos},\\
\text{where }\mathbf{W}^E&\in\mathbb{R}^{(P^2\cdot C)\times d},\mathbf{E}^{pos}\in\mathbb{R}^{(n+1)\times d}\\
I^p &\in \mathbb{R}^{n\times( P^2\cdot}, n = \frac{HW}{P^2}
\end{aligned}
```

For weight quantization, we quantize the
weights W^Q^, W^K^, W^V^
,W^O^,W^1^
,W^2^, as well as the
linear embedding W^E^

```tex
\begin{aligned}
\mathbf{A}_l&=\mathbf{Q}_l\mathbf{K}_l^{\mathrm{T}}=\mathbf{X}_l \mathbf{W}_l^Q\mathbf{W}_l^{\mathrm{K}^\mathrm{T}}\mathbf{X}_l^{\mathrm{T}},\\
\mathrm{MSA}(\mathbf{X}_l)&=\mathrm{Softmax}(\frac{1}{\sqrt{d}}\mathbf{x}_l)\mathbf{x}_l\mathbf{W}_l^V\cdot\mathbf{W}_l^O\\
\mathbf{Z}_l&=\mathrm{LN}(\mathbf{X}_l+\mathrm{MSA}(\mathbf{X}_l)),\\
\mathbf{X}_{l+1}&=\mathrm{LN}(\mathbf{Z}_l+\mathrm{MLP}(\mathbf{Z}_l))\\
\end{aligned}
```

Besides these weights, we also quantize the inputs of all linear
layers and matrix multiplication operations.
Following the methods in [22, 34], we **do not** quantize
the softmax operation and layer normalization, because the parameters contained in these operations
are negligible and quantizing them **may bring significant accuracy degradation**.

### Optimization for Post-Training Quantization

The
choice of quantization intervals is critical for quantization and **one popular option** is to use a uniform
quantization function, where the data range is **equally** split:
```tex
\Psi_\Delta(\mathbf{Y})=\text{Clip}(\text{Round}(\frac{\mathbf{Y}}\Delta),-2^{b-1},2^{b-1}-1)
```


#### Similarity-Aware Quantization for Linear Operation
Original one and quantized one are as follows:

```tex
\begin{aligned}

\mathbf{O}_l &=\mathbf{X}_l \mathbf{W}_l \\ 
\widehat{\mathbf{O}}_l&=\Psi_{\Delta_l^X}(\mathbf{X}_l)\Psi_{\Delta_l^W}(\mathbf{W}_l)\cdot\Delta_l^W\cdot\Delta_l^X\\

\end{aligned}
```

it can be seen that the
**quantization intervals** actually **control** the _clipping thresholds_ in quantization process, which affects
the **similarity** between original output feature maps and quantization feature maps to a great extent.

In the l-th transformer layer, the similarity-aware quantization can be
formulated as:

```tex
\begin{aligned}
&\max_{\Delta_l^W,\Delta_l^X}\frac1N\sum_{i=1}^N\Gamma(\mathbf{O}_l^i,\widehat{\mathbf{O}}_l^i)\quad s.t.\Delta_l^W,\Delta_l^X\in\mathbb{R}^+\\

&\Gamma(\widehat{\mathbf{O}},\mathbf{O})=\frac{\sum_{j=1}^m(\mathbf{O}_j-\overline{\mathbf{O}})(\widehat{\mathbf{O}}_j-\overline{\widehat{\mathbf{O}}})}{\sqrt{\sum_{j=1}^m(\mathbf{O}_j-\overline{\mathbf{O}})^2}\sqrt{\sum_{j=1}^m(\widehat{\mathbf{O}}_j-\overline{\widehat{\mathbf{O}}})^2}}.
\end{aligned}
```

#### Ranking-Aware Quantization for Self-Attention

The self-attention layer is the critical component of the transformer
since it can calculate the **global relevance** of the features,
which makes the
transformer unique from the convolutional neural networks.
For the calculation of self-attention
(Eq. 3), we empirically find that the **relative order of the attention map** has been **changed** after
quantization as shown in Fig 1, which could **cause a significant performance degradation**.
Thus, a
ranking loss is introduced to solve this problem during the quantization process:

```tex
\begin{aligned}
&\max_{\Delta_l^W,\Delta_l^X}\frac1N\sum_{i=1}^N\Gamma(\mathbf{O}_l^i,\widehat{\mathbf{O}}_l^i)-\gamma\cdot\mathcal{L}_{ranking} \quad s.t.\Delta_l^W,\Delta_l^X\in\mathbb{R}^+\\

&\mathcal{L}_{ranking}=\sum_{k=1}^{h}\sum_{i=1}^{w-1}\sum_{j=i+1}^{w}\Phi((\widehat{\mathbf{A}}_{ki}-\widehat{\mathbf{A}}_{kj})\cdot sign(\mathbf{A}_{ki}-\mathbf{A}_{kj}))\\
&\Phi(p)=(\theta-p)_{+} \text{ is hinge function with parameter θ }\\
\end{aligned}
```

To solve the above optimization problem, we present a simple but efficient alternative searching
method for the uniform quantization of transformer layers.

We alternatively fix \Delta^W^_l,\Delta^X^_l to make the other one be optimal.

Moreover, for fast convergence, They
are initialized in terms of **the maximum of weights or
inputs** respectively.
For the search space of them
, we linearly divide intervals of [α∆l
, β∆l
]
into C candidate options and conduct a simple search strategy on them

#### Bias Correction

Suppose the quantization error of
weights and inputs are defined as:
```tex
\begin{aligned}\epsilon^X&=\Psi_{\Delta^X}(\mathbf{X})\cdot\Delta^X-\mathbf{X},\\\epsilon^W&=\Psi_{\Delta^W}(\mathbf{W})\cdot\Delta^W-\mathbf{W}.\end{aligned}
```

If the **expectation of the error** for output is not zero, then the **mean** of the output will change.
This
**shift in distribution** may lead to detrimental behavior in the following layers.
We can correct this
change by seeing that:
```tex
\mathbb{E}[\widehat{\mathbf{O}}]=\mathbb{E}[\mathbf{O}]+\mathbb{E}[\epsilon^W\mathbf{X}]+\mathbb{E}[\epsilon^X\mathbf{W}]+\mathbb{E}[\epsilon^X\epsilon^W]
```

Thus, **subtracting** the **expected error** on the output **from the biased output** ensures that the mean for
each output unit is **preserved**.
For implementation, the expected error can be computed using the
**calibration data** and subtracted from the layer’s **bias parameter**, since the expected error vector has
the same shape as the layer’s output.

### Mixed-Precision Quantization for Vision Transformer

Considering the unique structure of transformer layer, we
assign **all the operations** in the **MSA or MLP modules** with **the same bit-width**.
This will also be
friendly to the hardware implementation since the weights and inputs are assigned with the same
bit-width.

The nuclear
norm is the **sum of singular values**, which represents the **data relevance** of the matrix. 

Inspired by the method in [10], we utilize a **Pareto frontier**
approach to determine the bit-width.
The main idea is to **sort** each _candidate bit-width configuration_
based on the **total second-order perturbation that they cause**, according to the following metric:

```tex
\Omega=\sum_{i=1}^L\Omega_i=\sum_{i=1}^L\sum_{j=1}^m\sigma_j(\mathbf{Y}_i)\cdot\|\widehat{\mathbf{Y}_i}-\mathbf{Y}_i\|_2^2.
```
\sigma_j is the j-th singular value on the diagonal.









