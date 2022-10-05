#!/usr/bin/env python
# coding: utf-8

# In[24]:


import tensorflow as tf
import tf_scatter


# In[25]:


class MeanSubtractionNorm(tf.keras.layers.Layer):
    def reset_parameters(self):
        pass

    def call(self, x:tf.Tensor, batch=None, dim_size= None) -> tf.Tensor:
        if batch is None:
            return x - tf.math.reduce_mean(x,axis=0, keepdims=True)

        mean = scatter_mean(x,batch,n_nodes=0)
        return x - mean[batch]


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


# In[ ]:




