# Joint Token Pruning and Squeezing Towards More Aggressive Compression of Vision Transformers

## Abstract

- Our
  quantitative experiments reveal that the impact of pruned
  tokens on performance should be noticeable.
-  To address
   this issue, we propose a novel joint Token Pruning &
   Squeezing module (TPS) for compressing vision transformers with higher efficiency.
  -  Firstly, TPS adopts pruning to get
     **the reserved and pruned subsets.** 
  - Secondly, TPS **squeezes**
     the information of **pruned tokens** into **partial reserved tokens** via the **unidirectional nearest-neighbor matching** and
     **similarity-based fusing** steps.

![image_20240304_131900.png](image_20240304_131900.png){thumbnail="true"}

## Method
### Motivation
The exclusive information from pruned tokens matters more
while the token pruning intensity grows.



###  Token Pruning
Here, we introduce two variants
of TPS: dTPS and eTPS, to cover both intra-block and inter-block token compression.

![image_20240304_141100.png](image_20240304_141100.png){thumbnail="true"}

dTPS adopts the **learnable token
score prediction head** from dynamicViT [25] and samples
the binary decision mask by Straight-Through **Gumbel-Softmax** [12] for differentiability; 

eTPS utilizes the **class
token attention values** to measure tokens’ importance as
EViT [16]. 

The
tokens are separated into two subsets, S
r
and S
p
, where the
reserved tokens are placed in S
r
and the pruned ones are
placed in S
p.

### Token Squeezing

Considering that the reserved
ones contribute the majority of correct predictions, we aim
to design a procedure that **retains** _most_ of the attentive tokens while **compressing** information from the rest, preserving
the model’s overall performance.

#### matching and fusing
Given the two subsets S
r
and S
p
, I
r
and
I
p
are the corresponding token indices of S
r
and S
p
. A
similarity matrix ci,j for all i ∈ I
p
and j ∈ I
r
represents
the interactions between the tokens for matching.

For each
pruned token xi ∈ S
p
, we find its nearest token x^
host
_∗ ∈ S
r
from the reserved token set S
r
as its host token:

```tex
{x}_*^{host}=\underset{{x}_j\in S^r}{\mathrm{argmax }} {c_{i,j}};
```

We then record the matching results in a mask matrix
M ∈ R ^{Np×Nr}

The similarity matrix is defined as:
```tex
c_{i,j}=\frac{{x}_i^T{x}_j}{\|{x}_i\|\|{x}_j\|},for~i\in I^p,j\in I^r.
```

**Fusing.**

Simply averaging tokens can lead to feature
dispersion because of discrepancies among the different
tokens.

As previously
mentioned, the fusing step _encompasses all tokens_ from two
subsets and is **controlled by the mask M** to ensure that only
**host tokens and pruned tokens** are mixed. 


Specifically, the reserved token xj is updated by fusing
the original feature and pruned tokens’ features as follows:
```tex
{y}_j=w_j{x}_j+\sum_{{x}_i\in S^p}w_i{x}_i,
```

The fusing weight wi depends on the mask value m_{i,j}
and similarity ci,j :
```tex
w_i=\frac{\exp(c_{i,j})m_{i,j}}{\sum_{{x}_i\in S^p}\exp(c_{i,j})m_{i,j}+\mathrm{e}}
```





