# Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks

## Abstract

- Specifically, the proposed SFP **enables the pruned filters to be updated** when training the model _after pruning_.
  - Larger model capacity(as they can be updated)
  - Less dependence on the pretrained model (Large capacity enables SFP to train from scratch and prune the model simultaneously)

## Introduction

### Efficient Architectures
- Reference [7] [ He et al. , 2016a ] Kaiming  He,  Xiangyu  Zhang,  Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR , 2016.
  > Residual learning partly eases the problem caused by deeper structure
- Inception: Reference [23] [ Szegedy et al. , 2015 ] Christian Szegedy, Wei Liu, Yangqing Jia,  Pierre  Sermanet,  Scott  Reed,  Dragomir  Anguelov, Dumitru  Erhan,  Vincent  Vanhoucke,  and  Andrew  Rabinovich. Going deeper with convolutions. In CVPR , 2015.
  - Normal Conv
  - ![image_20240119_202500.png](image_20240119_202500.png)
  - Depth-wise Conv 
  - ![image_20240119_202700.png](image_20240119_202700.png)
    > One Channel <-> One Filter
  - Point-wise Conv
  - ![image_20240119_203000.png](image_20240119_203000.png)
    > Input dim is the same as output dim.
  - Depth Separable (combined above two) (Explains why use point-wise)
  - ![image_20240119_211900.png](image_20240119_211900.png){thumbnail="true"}
  - Inception Module (1x1 conv introduced for dimension reduction)
  - ![image_20240120_102600.png](image_20240120_102600.png)
  - GoogLeNet
  - ![image_20240120_102700.png](image_20240120_102700.png)
    > DepthConcat means that concatenation along depth axis.
    > 
    > <a href="https://hacktildawn.com/2016/09/25/inception-modules-explained-and-implemented/"></a>

### Recent Efforts
- weight pruning
- filter pruning

Nevertheless, most of the previous works on filter pruning 
still suffer from the problems of 
- (1) the model capacity reduction and 
- (2) the dependence on pre-trained model.

## Related Works
Most previous works on accelerating CNNs can be roughly divided into three categories, namely, 
- matrix decomposition, 
- low-precision weights, 
- and pruning.
  - weight pruning
  - filter pruning
    - uses `L_1`-norm to select unimportant filters
    - introduces `L_1` regularization on the scaling factors in batch normalization (BN) layers as a **penalty term**, and prune channel with **small scaling factors** in BN layers.
    - a **Taylor expansion based pruning criterion** to approximate the change in the cost function induced by pruning.
    - adopts the **statistics information** from next layer to guide the importance evaluation of filters.


## Methodology

### Preliminaries

CNN parameterized as 
```tex
\{W(i) ∈ \mathbb{R}^{N_{i+1}×N_i×K×K} , 1 ≤ i ≤ L\}

```

Def:

```tex
W(i)
``` 
denotes a matrix of connection weights in the i-th layer. 
```tex
N_i
``` 
denotes the number of input channels for the i-th convolution layer. 
```tex
L
```
denotes the number of layers.

The shapes of input tensor U and output tensor V are `N_i × H_i × W_i` and `N_i+1 × H_i+1 × W_i+1`, respectively.

The convolutional operation of the i-th layer can be written as:

```tex 
V_{i,j} = F_{i,j} ∗ U \text{ for } 1 ≤ j ≤ N_{i+1},
```

where `F_{i,j} ∈ R^{N_i×K×K}` represents the j-th filter of the i-th layer.

Let us assume the **pruning rate** of SFP is `P_i` for the i-th layer.

### Soft Filter Pruning (SFP)

The key is to **keep updating the pruned filters** in the **training stage**

After each epoch, **the L2-norm of all filters** are computed for each weighted layer and used as the criterion of our filter selection strategy.

Then we will prune the selected filters by **setting the corresponding filter weights as zero**, which is **followed by next training epoch**. Finally, the original deep CNNs are pruned into a compact and efficient model.

![image_20240121_161600.png](image_20240121_161600.png)

#### Filter selection

We use the L_p-norm to evaluate the importance of each filter as Eq. (2)

#### Filter Pruning
In the filter pruning step, we simply prune all the weighted layers at the same time. In this way, we can **prune each filter in parallel**, which would cost negligible computation time.

#### Reconstruction

![image_20240121_183100.png](image_20240121_183100.png){thumbnail="true"}

#### Obtaining Compact Model
Finally, a compact model
```tex 
{W^∗(i) ∈ \mathbb{R}^{N_{i+1}(1−P_i)×N_i(1−P_{i−1})×K×K} }
``` 
is obtained.




