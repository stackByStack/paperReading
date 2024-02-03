# RepViT: Revisiting Mobile CNN From ViT Perspective

<show-structure for="chapter,procedure" depth="3"/>

## Abstract
- the **architectural disparities** between
  lightweight ViTs and lightweight CNNs have not been adequately examined.
- We incrementally enhance the **mobile friendliness** of a standard lightweight CNN, 
  specifically MobileNetV3, by integrating the **efficient architectural choices**
  of lightweight ViTs
- This ends up with a new family of
  pure lightweight CNNs, namely RepViT

## Introduction
### Efficient Design Principles

- <a href="https://arxiv.org/abs/1704.04861">a streamlined architecture using depth-wise separable convolutions </a>
- <a href="https://arxiv.org/abs/1801.04381">inverted residual bottleneck</a>
- <a href="https://arxiv.org/abs/1807.11164">channel split and channel shuffle</a>

![image_20240201_110200.png](image_20240201_110200.png){thumbnail="true"}

### ViT
<a href="https://arxiv.org/abs/2010.11929">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a>

a pure transformer applied directly to **sequences of image patches** can perform
very well on image classification tasks.

We split an image
into patches and provide the sequence of **linear embeddings** of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application. We train
the model on image classification in supervised fashion.

![image_20240201.png](image_20240201.png)

#### Image Classification
##### Swin Transformer
<a href="https://arxiv.org/pdf/2103.14030.pdf">Swin Transformer</a>

![image_20240201_173400.png](image_20240201_173400.png)

![image_20240201_174100.png](image_20240201_174100.png){thumbnail="true"}

![image_20240201_180600.png](image_20240201_180600.png)

##### Pyramid Vision Transformer
<a href="https://arxiv.org/pdf/2102.12122.pdf">Pyramid Vision Transformer</a>

![image_20240201_185400.png](image_20240201_185400.png)

#### semantic segmentation

##### Masked-attention Mask Transformer for Universal Image Segmentation
<a href="https://arxiv.org/pdf/2112.01527.pdf"></a>

- Unify pan-optic, instance, semantic segmentation
- ![image_20240201_225400.png](image_20240201_225400.png)

Masks are learned through training.

##### SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

<a href="https://arxiv.org/pdf/2105.15203.pdf"></a>

![image_20240202_093400.png](image_20240202_093400.png)

#### Object Detection
##### End-to-End Object Detection with Transformers

![image_20240202_102800.png](image_20240202_102800.png)

##### MViTv2: Improved Multi-scale Vision Transformers for Classification and Detection
![image_20240202_110000.png](image_20240202_110000.png)

![image_20240202_110300.png](image_20240202_110300.png)

### Enhancing Computational Efficiency of Vision Transformers for Mobile Devices through Effective Design Principles

#### Mobile-Former: Bridging MobileNet and Transformer
![image_20240202_205700.png](image_20240202_205700.png)

![image_20240202_210500.png](image_20240202_210500.png)
#### Rethinking vision transformers for mobilenet size and speed.

- proposed EfficientFormerV2
- Token Mixers
- Search Space Refinement
- MHSA Improvements
- Attention on Higher Resolution
- Dual-Path Attention Down sampling (CNN + Attention Parallel Computation and Sum)

#### MOBILEVIT: LIGHT-WEIGHT, GENERAL-PURPOSE, AND MOBILE-FRIENDLY VISION TRANSFORMER
![image_20240202_221300.png](image_20240202_221300.png)
- No need for patch embedding

#### EdgeViTs: Competing Light-weight CNNs on Mobile Devices with Vision Transformers
![image_20240203_1005.png](image_20240203_1005.png)

![image_20240203_1007.png](image_20240203_1007.png)

#### Separable Self-attention for Mobile Vision Transformers: MobileViTv2

![image_20240203_103900.png](image_20240203_103900.png){thumbnail="true"}

![image_20240203_105900.png](image_20240203_105900.png)

### Summary 
- In this work, we
revisit the design of lightweight CNNs **by incorporating
the architectural choices** of lightweight ViTs. 
- Our research
aims to narrow the divide between lightweight CNNs and
lightweight ViTs, and highlight the potential of **the former
for employment on mobile devices** compared to the latter.

we begin with a standard lightweight CNN, i.e., MobileNetV3-
L. We gradually “modernize” its architecture by incorporating the efficient architectural designs of lightweight
ViTs 

RepViT has a MetaFormer 
structure, but is composed entirely of convolutions.

![Generic Architecture](image_20240203_171100.png)

## Methodology
### Preliminary
-  We utilize the iPhone
12 as the test device and Core ML Tools as the compiler.

- We measure the actual **on-device latency** for models
as the benchmark metric.
- employ GeLU activations in the MobileNetV3-L model

![image_20240203_232700.png](image_20240203_232700.png)

<a href="Data-Augmentation.md">Details about data-augmentation trick mentioned above</a>

### Block Design
![image_20240204_000500.png](image_20240204_000500.png){thumbnail="true"}

![image_20240204_000600.png](image_20240204_000600.png)

1x1 expansion is typically used to increase the number of channels in the feature maps,
while the 1x1 projection convolution is used for dimensionality reduction.
The combination of them is **channel mixer**.

And the depth-wise convolution is **token mixer**.

The squeeze and excitation module is also
moved up to be placed after the depth-wise filters, as it **depends on spatial information interaction**.
















