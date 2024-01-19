# Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks

## Abstract

- Specifically, the proposed SFP **enables the pruned filters to be updated** when training the model _after pruning_.
  - Larger model capacity(as they can be updated)
  - Less dependence on the pretrained model (Large capacity enables SFP to train from scratch and prune the model simultaneously)

## Introduction

- Efficient Architectures
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
    - ![image_20240119_211900.png](image_20240119_211900.png)
    - 