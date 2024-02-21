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

## Global Dynamic Pruning
### The Proposed Pruning Scheme
> Our goal is to globally prune redundant filters
> 
> We introduce a
> global mask to temporally mask out unsalient filters in each
> iteration during training.
> 

Then, conventional convolution layer computation could be rewritten like this:

```tex
\mathbf{Z}_{l}^{*}=f\Big(\mathbf{Z}_{l-1}^{*}\times(\mathbf{W}_{l}^{*}\odot\mathbf{m}_{l})\Big),\quad s.t. \quad l=1,2,\cdots,L.
```
```tex
m^k_l = 1
```

if the k-th filter is salient in the l-th layer, and 0 otherwise. 
```tex
\odot
```
denotes the Khatri-Rao product operator.

We propose to solve the following optimization problem:
```tex
\begin{array}{rl}\min&\mathcal{L}\big(\mathcal{Y},g(\mathcal{X};\mathcal{W}^*,\mathbf{m})\big)\\
s.t.&\mathbf{m}=h(\mathcal{W}^*)\\&\left\|\mathbf{m}\right\|_0\leq\beta\sum_{l=1}^LC_l,\end{array}
```

The problem is NP-hard, because of
the || · ||_0 operator

h(·) is a global discriminative function to determine the saliency values of filters,
which depends on the prior knowledge of W^∗ .

The output
entry of function h(·) is binary, i.e., to be 1 if the corresponding filter is salient, and 0 otherwise.

### Solver
Since every filter has a mask, we
update W∗
as below:

```tex
\mathbf{W}_{l}^{*}=\mathbf{W}_{l}^{*}-\eta\frac{\partial\mathcal{L}(\mathcal{Y},g(\mathcal{X};\mathcal{W}^{*},\mathbf{m}))}{\partial(\mathbf{W}_{l}^{*}\odot\mathbf{m}_{l})},l=1,\cdots,L,
```

![image_20240221_165000.png](image_20240221_165000.png)

> To accelerate the convergence of Algorithm 1, we set a low
> frequency for the global mask updating, which is controlled
> by the threshold e
> 
> And the global mask is not updated when
> the network is in the warm-up phase
>

> Why?
> 
> With a large loss of the network in the unstable warm-up phase, frequently updating the
> global mask cannot provide useful information to guide the
> network pruning.

### The Global Mask
```tex
\begin{aligned}
\left|\Delta\mathcal{L}(\mathcal{Y},g(\mathcal{X};\mathbf{W}_{l}^{k*}))\right|=& \left|\mathcal{L}(\mathcal{Y},g(\mathcal{X};\mathbf{W}_{l}^{k*}=\mathbf{0})\right)  \\
&-\left.\mathcal{L}(\mathcal{Y},g(\mathcal{X};\mathbf{W}^{*}))\right|,
\end{aligned}
```

simplified as

```tex
\left|\Delta\mathcal{L}(\mathbf{W}_l^{k*})\right|=\left|\mathcal{L}(\mathcal{D},\mathbf{W}_l^{k*}=\mathbf{0})-\mathcal{L}(\mathcal{D},\mathbf{W}^*)\right|.
```

Then,

```tex
\Big|\Delta\mathcal{L}(\mathbf{W}_{l}^{k*})\Big|\approx\Big|\frac{\partial\mathcal{L}(\mathcal{D},\mathbf{W}^{*})}{\partial\mathbf{W}_{l}^{k*}}\mathbf{W}_{l}^{k*}\Big|.
```

Since the filter W^\{k∗\} \_l

is a d^2C_{l−1}-dimensional vector, we construct
a function to measure the saliency score of a filter.

```tex
f_T(\mathbf{W}_l^{k*})=\Big|\sum_{r=1}^{d^2C_{l-1}}\frac{\partial\mathcal{L}(\mathcal{D},\mathbf{W}^*)}{\partial\mathbf{W}_{l,r}^{k*}}\mathbf{W}_{l,r}^{k*}\Big|
```




















