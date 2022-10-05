#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tf_scatter import scatter_mean


# In[2]:


class PairNorm(tf.keras.layers.Layer):
   
    def __init__(self, scale: float = 1., scale_individually: bool = False,
                 eps: float = 1e-5):
        super().__init__()

        self.scale = scale
        self.scale_individually = scale_individually
        self.eps = eps


    def call(self, x: tf.Tensor, batch= None) -> tf.Tensor:
        
        scale = self.scale

        if batch is None:
             
            x = x - tf.math.reduce_mean(x ,axis = 0, keepdims= True)


            if not self.scale_individually:
                
                return scale * x / tf.math.sqrt(self.eps + tf.math.reduce_mean(tf.math.reduce_sum(tf.math.pow (x,2),-1))) 

            else:
                 
                return scale * x / (self.eps + tf.norm(x, ord=2, axis=-1, keepdims=True))

        else:
            
            mean = scatter_mean(x , batch,n_nodes=0)
            
            x = x - tf.gather(mean, indices= batch, axis=0).numpy()


            if not self.scale_individually:
                
                return scale * x /  tf.math.sqrt(self.eps + scatter_mean(x.tf.math.pow(2).reduce_sum(-1, keepdims=True), batch, n_nodes=0).tf.gather(indices= batch, axis=0))  


            else:
                return scale * x / (self.eps + tf.norm(x, ord=2, axis=-1, keepdims=True))


    def __repr__(self):
        return f'{self.__class__.__name__}()'


# In[ ]:




