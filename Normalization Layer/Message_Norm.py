#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


class MessageNorm(tf.keras.layers.Layer):
 
    def __init__(self, learn_scale: bool = False):
        super().__init__()
     

        self.scale = tf.Variable(tf.constant([1.0]), trainable=learn_scale)

    def reset_parameters(self):
         self.scale = tf.fill((1,) , 1.0 , name ="r")
         return self.scale
        




    def call(self, x: tf.Tensor, msg: tf.Tensor, p: int = 2) -> tf.Tensor:
       
        
        msg = tf.keras.utils.normalize(msg, axis=-1, order=p)

        x_norm = tf.norm(x, ord=p, axis=-1, keepdims=True)
        return msg * x_norm * self.scale


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f'(learn_scale={self.scale.requires_grad})')


# In[ ]:




