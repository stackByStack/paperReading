# Group Shuffle

## Group Conv
![image_20240122_115500.png](image_20240122_115500.png)

> Filters also need to be grouped correspondingly.
> 

## Channel Shuffle

Details could be checked here.

<a href="https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/shufflenetv1.py"></a>

```Python
def channel_shuffle(self, x: Tensor) -> Tensor:
    batch_size, num_channels, height, width = x.shape

    group_channels = num_channels // self.group
    x = ops.reshape(x, (batch_size, group_channels, self.group, height, width))
    x = ops.transpose(x, (0, 2, 1, 3, 4))
    x = ops.reshape(x, (batch_size, num_channels, height, width))
    return x
```

**Memory Layout**

123 456 789 (3 groups)

reordered as

147 258 369

assuming `num_channels % self.group != 0` here.