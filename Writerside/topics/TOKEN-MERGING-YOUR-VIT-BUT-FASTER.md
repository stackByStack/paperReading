# TOKEN MERGING: YOUR VIT BUT FASTER

## Abstract
- Token Merging(ToMe) gradually **combines similar
  tokens** in a transformer using a general and light-weight matching algorithm that
  is as fast as pruning while being more accurate.

## Introduction
Yet, token
pruning has several disadvantages: 
- the information loss from pruning limits how many tokens you
can reasonably reduce; 
- current methods require **re-training** the model to be effective (some with extra
parameters); 
- most cannot be applied to speed up training; and several prune different numbers of
tokens depending on the input content, **making batched inference _infeasible_.**















