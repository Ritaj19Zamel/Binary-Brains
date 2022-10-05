#!/usr/bin/env python
# coding: utf-8

# In[73]:


import tensorflow as tf
from tensorflow import reshape
import tf_degree
import tf_scatter


# In[74]:


def tf_index_select(input_, dim, indices):
    """
    input_(tensor): input tensor
    dim(int): dimension
    indices(list): selected indices list
    """
    shape = input_.get_shape().as_list()
    if dim == -1:
        dim = len(shape)-1
    shape[dim] = 1
    
    tmp = []
    for idx in indices:
        begin = [0]*len(shape)
        begin[dim] = idx
        tmp.append(tf.slice(input_, begin, shape))
    res = tf.concat(tmp, axis=dim)
    
    return res


# In[75]:


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, in_channels:int, eps:float= 1e-5, affine: bool = True, mode: str = 'graph'):
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.mode = mode
        if affine:
            self.weight = tf.Variable(tf.constant([in_channels]))
            self.bias =  tf.Variable(tf.constant([in_channels]))
        else:
            layer.add_weight('weight', None)
            layer.add_weight('bias', None)
        self.reset_parameter()
    def reset_parameter(self):
        tf.ones(self.weight)
        tf.zeros(self.bias)
    def forward(self, x:tf.Tensor, batch = None) ->tf.Tensor:
        if self.mode == 'graph':
            if batch is None:
                x = x - tf.math.reduce_mean(x)
                out = x / (tf.math.reduce_std(x,  keepdims=False) + self.eps)
                

            else:
                batch_size = int(batch.max()) + 1
                norm = tf_degree (batch, batch_size, dtype=x.dtype).clamp_(min=1)
                norm = norm.mul_(x.size(-1)).view(-1, 1)
                mean =tf_scatter(x, batch, dim=0, dim_size=batch_size,
                                reduce='add').sum(dim=-1, keepdim=True) / norm

                x = x - mean.index_select(0, batch)

                var = tf_scatter(x * x, batch, dim=0, dim_size=batch_size,
                                reduce='add').sum(dim=-1, keepdim=True)
                var = var / norm

                out = x / tf_index_select(tf.math.sqrt(var + self.eps),0, batch)

            if self.weight is not None and self.bias is not None:
                
                out = out * tf.cast(self.weight,dtype=tf.float32) +tf.cast(self.bias,dtype=tf.float32)

            return out
        if self.mode == 'node':
            return tf.keras.layers.Normalization(x, (self.in_channels, ), self.weight,
                                self.bias, self.eps)

        raise ValueError(f"Unknow normalization mode: {self.mode}")
    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'mode={self.mode})')


# In[ ]:




