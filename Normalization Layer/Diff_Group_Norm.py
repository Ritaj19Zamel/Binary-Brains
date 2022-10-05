#!/usr/bin/env python
# coding: utf-8

# In[350]:


import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages
import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers.core import Layer
import tensorflow.compat.v2 as tf
import keras
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.dtensor import utils
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import control_flow_util
from keras.utils import tf_utils


from tensorflow.python.ops.control_flow_ops import (
    get_enclosing_xla_context,
)
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export


# In[351]:


class LocalResponseNorm(Layer):

    def __init__(
        self,
        depth_radius=None,
        bias=None,
        alpha=None,
        beta=None,
        name=None,  #'lrn',
    ):
        # super(LocalResponseNorm, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

        logging.info(
            "LocalResponseNorm %s: depth_radius: %s, bias: %s, alpha: %s, beta: %s" %
            (self.name, str(depth_radius), str(bias), str(alpha), str(beta))
        )

    def build(self, inputs):
        pass

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a 4D output shape.
        """
        outputs = tf.nn.lrn(inputs, depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta)
        return outputs



def _to_channel_first_bias(b):
    """Reshape [c] to [c, 1, 1]."""
    channel_size = int(b.shape[0])
    new_shape = (channel_size, 1, 1)
    # new_shape = [-1, 1, 1]  # doesn't work with tensorRT
    return tf.reshape(b, new_shape)


def _bias_scale(x, b, data_format):
    """The multiplication counter part of tf.nn.bias_add."""
    if data_format == 'NHWC':
        return x * b
    elif data_format == 'NCHW':
        return x * b
    else:
        raise ValueError('invalid data_format: %s' % data_format)


def _bias_add(x, b, data_format):
    """Alternative implementation of tf.nn.bias_add which is compatiable with tensorRT."""
    if data_format == 'NHWC':
        return tf.add(x, b)
    elif data_format == 'NCHW':
        return tf.add(x, b)
    else:
        raise ValueError('invalid data_format: %s' % data_format)


def _compute_shape(tensors):
    if isinstance(tensors, list):
        shape_mem = [t.get_shape().as_list() for t in tensors]
    else:
        shape_mem = tensors.get_shape().as_list()
    return shape_mem


def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, data_format, name=None):
    """Data Format aware version of tf.nn.batch_normalization."""
    if data_format == 'channels_last':
        mean = tf.reshape(mean, [1] * (len(x.shape) - 1) + [-1])
        variance = tf.reshape(variance, [1] * (len(x.shape) - 1) + [-1])
        offset = tf.reshape(offset, [1] * (len(x.shape) - 1) + [-1])
        scale = tf.reshape(scale, [1] * (len(x.shape) - 1) + [-1])
    elif data_format == 'channels_first':
        mean = tf.reshape(mean, [1] + [-1] + [1] * (len(x.shape) - 2))
        variance = tf.reshape(variance, [1] + [-1] + [1] * (len(x.shape) - 2))
        offset = tf.reshape(offset, [1] + [-1] + [1] * (len(x.shape) - 2))
        scale = tf.reshape(scale, [1] + [-1] + [1] * (len(x.shape) - 2))
    else:
        raise ValueError('invalid data_format: %s' % data_format)

    with ops.name_scope(name, 'batchnorm', [x, mean, variance, scale, offset]):
        inv = math_ops.rsqrt(variance + variance_epsilon)
        if scale is not None:
            inv *= scale

        a = math_ops.cast(inv, x.dtype)
        b = math_ops.cast(offset - mean * inv if offset is not None else -mean * inv, x.dtype)

        # Return a * x + b with customized data_format.
        # Currently TF doesn't have bias_scale, and tensorRT has bug in converting tf.nn.bias_add
        # So we reimplemted them to allow make the model work with tensorRT.
        # See https://github.com/tensorlayer/openpose-plus/issues/75 for more details.
        df = {'channels_first': 'NCHW', 'channels_last': 'NHWC'}
        return _bias_add(_bias_scale(x, a, df[data_format]), b, df[data_format])



class BatchNorm(Layer):
    
    def __init__(
        self,
        decay=0.9,
        epsilon=0.00001,
        act=None,
        is_train=False,
        beta_init=tl.initializers.zeros(),
        gamma_init=tl.initializers.random_normal(mean=1.0, stddev=0.002),
        moving_mean_init=tl.initializers.zeros(),
        moving_var_init=tl.initializers.zeros(),
        num_features=None,
        data_format='channels_last',
        name=None,
    ):
        super(BatchNorm, self).__init__(name=name, act=act)
        self.decay = decay
        self.epsilon = epsilon
        self.data_format = data_format
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.moving_mean_init = moving_mean_init
        self.moving_var_init = moving_var_init
        self.num_features = num_features

        self.axes = None

        if num_features is not None:
            self.build(None)
            self._built = True

        if self.decay < 0.0 or 1.0 < self.decay:
            raise ValueError("decay should be between 0 to 1")

        logging.info(
            "BatchNorm %s: decay: %f epsilon: %f act: %s is_train: %s" %
            (self.name, decay, epsilon, self.act.__name__ if self.act is not None else 'No Activation', is_train)
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(num_features={num_features}, decay={decay}' ', epsilon={epsilon}')
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name="{name}"'
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def _get_param_shape(self, inputs_shape):
        if self.data_format == 'channels_last':
            axis = -1
        elif self.data_format == 'channels_first':
            axis = 1
        else:
            raise ValueError('data_format should be either %s or %s' % ('channels_last', 'channels_first'))

        channels = inputs_shape[axis]
        params_shape = [channels]

        return params_shape

    def _check_input_shape(self, inputs):
        inputs_shape = _compute_shape(inputs)
        if len(inputs_shape) <= 1:
            raise ValueError('expected input at least 2D, but got {}D input'.format(inputs.ndim))

    def build(self, inputs_shape):
        params_shape = [self.num_features] if self.num_features is not None else self._get_param_shape(inputs_shape)

        self.beta, self.gamma = None, None
        if self.beta_init:
            self.beta = self._get_weights("beta", shape=params_shape, init=self.beta_init)

        if self.gamma_init:
            self.gamma = self._get_weights("gamma", shape=params_shape, init=self.gamma_init)

        self.moving_mean = self._get_weights(
            "moving_mean", shape=params_shape, init=self.moving_mean_init, trainable=False
        )
        self.moving_var = self._get_weights(
            "moving_var", shape=params_shape, init=self.moving_var_init, trainable=False
        )

    def forward(self, inputs):
        self._check_input_shape(inputs)

        self.channel_axis = len(inputs.shape) - 1 if self.data_format == 'channels_last' else 1
        if self.axes is None:
            self.axes = [i for i in range(len(inputs.shape)) if i != self.channel_axis]

        mean, var = tf.nn.moments(inputs, self.axes, keepdims=False)
        if self.is_train:
            # update moving_mean and moving_var
            self.moving_mean = moving_averages.assign_moving_average(
                self.moving_mean, mean, self.decay, zero_debias=False
            )
            self.moving_var = moving_averages.assign_moving_average(self.moving_var, var, self.decay, zero_debias=False)
            outputs = batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon, self.data_format)
        else:
            outputs = batch_normalization(
                inputs, self.moving_mean, self.moving_var, self.beta, self.gamma, self.epsilon, self.data_format
            )
        if self.act:
            outputs = self.act(outputs)
        return outputs



class BatchNorm1d(BatchNorm):
    

    def _check_input_shape(self, inputs):
        inputs_shape = _compute_shape(inputs)
        if len(inputs_shape) != 2 and len(inputs_shape) != 3:
            raise ValueError('expected input to be 2D or 3D, but got {}D input'.format(inputs.ndim))


# In[364]:


class DiffGroupNorm(tf.keras.layers.Layer):
        
    def __init__(self, in_channels, groups, lamda=0.01, epsilon=1e-5, decay=0.1):
        super().__init__()

        self.in_channels = in_channels
        self.groups = groups
        self.lamda = lamda

        self.lin = tf.keras.layers.Dense(groups, input_shape=(in_channels,), activation='softmax')
        self.norm = tl.layers.BatchNorm1d(num_features=groups * in_channels)

        
    def reset_parameters(self):
            self.lin.reset_parameters()
            self.norm.reset_parameters()
            
    def call(self, x: tf.Tensor) -> tf.Tensor:
        F, G = self.in_channels, self.groups
        s = self.lin(x)
        out = tf.expand_dims(s,-1) * tf.expand_dims(x,-2)
        reshap_out = tf.reshape(out,[-1,G*F])
        out_norm = self.norm(reshap_out)
        out = tf.reduce_sum(tf.reshape(out_norm,[-1, G, F]), axis = -2)
        return x + self.lamda * out
    @staticmethod
    def group_distance_ratio(x: tf.Tensor, y: tf.Tensor, epsilon: float = 1e-5) -> float:
        def cdist(a, b):
            return tf.sqrt(reduce_sum(tf.square(tf.expand_dims(a, 0) - tf.expand_dims(b, 1))))
        num_classes = int(tf.reduce_max(y)) + 1
        numerator = 0.
        for i in range(num_classes):
            mask = y == i
            dist= cdist(tf.expand_dims(x[mask],0), tf.expand_dims(x[mask],0))
            numerator += (1/tf.size(dist)) * float(reduce_sum(dist))
        numerator *= 1 / (num_classes - 1)**2

        denominator = 0.
        for i in range(num_classes):
            mask = y == i
            dist= cdist(tf.expand_dims(x[mask],0), tf.expand_dims(x[mask],0))
            denominator += (1/tf.size(dist)) * float(reduce_sum(dist))
        denominator *= 1 / num_classes

        return numerator / (denominator + epsilon)
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'groups={self.groups})')


# In[367]:


group =DiffGroupNorm(2,4)
group()


# In[368]:


x = tf.constant([[-0.21207808 ,1.0188377],
 [ 1.0870069   ,0.55926543],
 [-2.6653123  ,-1.4276485 ],
 [-1.1202446   ,0.43419424]])
group=DiffGroupNorm(2,4)
group(x)

