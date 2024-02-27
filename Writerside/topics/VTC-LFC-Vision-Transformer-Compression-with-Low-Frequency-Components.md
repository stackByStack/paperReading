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













