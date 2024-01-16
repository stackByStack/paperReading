# BERT

## TLDR
<tldr>
BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on <control>both left and right context in all layers</control>
</tldr>

## Intro

### Downstream Task Strategies

- feature based 
  - The feature-based approach, such as ELMo (Peters et al., 2018a), uses task-specific architectures that **include the pre-trained representations as additional features**.
- fine-tuning
  - Generative Pre-trained Transformer (OpenAI GPT) (Radford et al., 2018), introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning all pre-trained parameters.

> the emphasis on minimal task-specific parameters indicates a preference for making targeted and focused modifications to **the existing parameters** rather than introducing an abundance of new ones

## Related Work

### Unsupervised Feature-based Approaches
Word -> Sentence -> Context

### Unsupervised Fine-tuning Approaches
More recently, sentence or document encoders which produce **contextual token representations** have been pre-trained from unlabeled text and **fine-tuned** for a supervised downstream task


## BERT
- There are two steps in our framework: pre-training and fine-tuning.
- For **fine-tuning**, the BERT model is first **initialized with the pre-trained parameters**, and **all the parameters** are fine-tuned using labeled data from the downstream tasks.

In this work, we denote **the number of layers** (i.e., Transformer blocks) as L, **the hidden size** as H, and the number of **self-attention heads** as A. We primarily report results on two model sizes: BERT_BASE (L=12, H=768, A=12, Total Parameters=110M) and BERT_LARGE (L=24, H=1024, A=16, Total Parameters=340M).

> In all cases we set the feed-forward/filter size to be 4H, i.e., 3072 for the H = 768 and 4096 for the H = 1024.
>

**Input/Output Representations**

To make BERT handle a variety of down-stream tasks, our input representation is able to unambiguously represent **both a single sentence and a pair of sentences** (e.g., 〈 Question, Answer 〉) in one token sequence.

![image_20240116_101300.png](image_20240116_101300.png)

As shown in Figure 1, we denote input embedding as E, the final hidden vector of the special \[CLS\] token as 
```tex 
C ∈ \mathbb{R}^H
``` 
, and the final hidden vector for the ith input token as 
```tex
T_i ∈ \mathbb{R}^H
```

### 3.1 Pre-training BERT

**Task #1: Masked LM**

> bidirectional conditioning would allow each word to indirectly “see itself”
>

In order to train a deep bidirectional representation, we simply **mask** some percentage of the input tokens at random, and then **predict** those masked tokens.

**Task #2: Next Sentence Prediction (NSP)**


![image_20240116_105300.png](image_20240116_105300.png)

### 3.2 Fine-tuning
<tldr>
For each task, we simply plug in the task-specific inputs and outputs into BERT and fine-tune all the parameters end-to-end.
</tldr>