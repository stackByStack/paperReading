# ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting

## Abstract

- We propose to re-parameterize a CNN into the remembering parts and forgetting parts, 
  - where the former learn to maintain the performance 
  - and the latter learn to prune
- Via training with regular **SGD** on the **former** but **a novel update rule with penalty gradients** on the latter, we realize **structured sparsity**.
- Then we equivalently **merge** the remembering and forgetting parts into the original architecture with narrower layers. 
- In this sense, ResRep can be viewed as a successful application of _Structural_ **Re-parameterization**

