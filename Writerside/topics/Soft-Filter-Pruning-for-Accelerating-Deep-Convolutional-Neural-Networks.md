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





