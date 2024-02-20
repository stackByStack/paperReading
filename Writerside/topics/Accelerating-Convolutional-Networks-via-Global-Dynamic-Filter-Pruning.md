# Accelerating Convolutional Networks via Global &amp; Dynamic Filter Pruning

## Abstract

- most approaches tend to prune filters in a **layer-wise fixed manner**, which is **incapable of dynamically recovering** the _previously removed_ filter, as well
as jointly optimize the pruned network **_across layers_**.
-  In this paper, we propose a novel global & dynamic pruning (GDP) scheme to prune redundant
   filters for CNN acceleration.
  - In particular, GDP
    first globally prunes the unsalient filters across all
    layers by proposing a global discriminative function based on prior knowledge of each filter. 
  - Second, it dynamically updates the filter saliency all
    over the pruned sparse network, and then recovers the mistakenly pruned filter, followed by a retraining phase to improve the model accuracy.
- Specially, we effectively solve the corresponding **non-convex optimization problem** of the proposed GDP
  via stochastic gradient descent with _greedy alternative updating._ 

















