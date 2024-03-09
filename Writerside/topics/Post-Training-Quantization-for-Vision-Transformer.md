# Post-Training Quantization for Vision Transformer

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
available to optimize the quantization model and the training costs for edge devices are intolerable.
> 
> 


### Our Contributions
In this paper, we study the post-training quantization method for vision transformer models with
mixed-precision for higher compression and speed-up ratios. 

The quantized process in the transformer
is formulated as an optimization problem for finding the optimal quantization intervals. Specially,
our goal is to maximize the **similarity** between the **full-precision and quantized outputs** in vision
transformers. 















