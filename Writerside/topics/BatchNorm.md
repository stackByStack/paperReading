# BatchNorm

<a href="https://en.wikipedia.org/wiki/Batch_normalization"></a>

Each layer of a neural network has inputs with a corresponding **distribution**, which is affected during the training process by the randomness in the parameter initialization and the randomness in the input data. The effect of these sources of randomness on the distribution of the inputs to internal layers during training is described as **internal co-variate shift**.

Batch normalization was initially proposed to **mitigate** internal co-variate shift.

## Transformation

Let us use B to denote a mini-batch of size m of the entire training set. The empirical mean and variance of B could thus be denoted as

![image_20240120_135600.png](image_20240120_135600.png)

Normalized as

![image_20240120_135700.png](image_20240120_135700.png)

![image_20240120_141100.png](image_20240120_141100.png)

## BP
How to update 
```tex 
\gamma \text{ and } \beta
```
![image_20240120_142200.png](image_20240120_142200.png)

## Inference

![image_20240120_142600.png](image_20240120_142600.png)

