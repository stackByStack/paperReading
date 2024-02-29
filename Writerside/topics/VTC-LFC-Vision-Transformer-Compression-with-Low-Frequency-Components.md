# VTC-LFC: Vision Transformer Compression with Low-Frequency Components

## Abstract
The compression only
in the spatial domain suffers from a dramatic performance drop without fine-tuning and is **not robust** to noise.

Because the noise in the spatial domain can easily
**confuse the pruning criteria**, leading to some parameters/channels being pruned
incorrectly.

Inspired by recent findings that self-attention is a **low-pass filter** and
low-frequency signals/components are more informative to ViTs, this paper proposes compressing ViTs with **low-frequency components**.

Two metrics named
**low-frequency sensitivity** (LFS) and **low-frequency energy** (LFE) are proposed
for better channel pruning and token pruning.

## Introduction
Token pruning based on low-frequency energy:
Token compression/sampling aims to select the informative tokens that store **more useful information**.

The _popular methods_ dynamically select those tokens with high correlation to other tokens (e.g. the
CLS token) as the informative tokens.
However, it may be sub-optimal because the selected tokens
tend to be similar to each other, and the information included in the token itself has been neglected to
some extent.











