# MobileNet v1-v3

MobileNet, a series of TensorFlow-based computer vision models, has evolved through various versions, each offering its unique specializations and improvements. Here are the key distinctions among them:

1. MobileNet V1: The first version of MobileNet, introduced in 2017, pioneered the use of **depth-wise separable convolutions** to build lightweight deep neural networks. It was known for its speed, being 10x faster and smaller than its predecessor, VGG16.

2. MobileNet V2: This version was built on the foundation of V1 but introduced significant improvements. The primary advancement was the addition of **inverted residuals and linear bottlenecks**, critically enhancing the model's performance without increasing its size. This allowed the model to retain its lightweight characteristic while improving accuracy. 

<a href="https://zhuanlan.zhihu.com/p/98874284">Details about inverted residuals and linear bottlenecks</a>

3. MobileNet V3: The concepts used in MobileNet V3 are an amalgamation of the innovations from the previous versions and novel technology advancements like **hardware-aware network architecture search (NAS)**. This version significantly improves the model’s performance in both speed and accuracy.

<a href="https://zhuanlan.zhihu.com/p/323346888">Details about Mob v3</a>
