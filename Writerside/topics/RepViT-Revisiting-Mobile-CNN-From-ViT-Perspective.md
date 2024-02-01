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














