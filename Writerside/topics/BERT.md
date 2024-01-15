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
