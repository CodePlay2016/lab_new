#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 22:56:04 2017

@author: codeplay2017
"""
# version1
#def batch_normalization(data, out_size, axes=[0], epsilon=0.001):
#    # do the batch normalization
#        # calculate the mean and variance of 'conv_Wx_Plus_b1'
#        # axes=[0] means it calculate in the dimension of batch_size
#    mean, var = tf.nn.moments(data, axes=[0])
#        # the two params below will be auto-adjusted by tensorflow
#    scale = tf.Variable(tf.ones([out_size]))
#    shift = tf.Variable(tf.zeros([out_size]))
#    return tf.nn.batch_normalization(data, mean, var, shift, scale, epsilon)

# version2
## this is a simpler version of Tensorflow's 'official' version. See:
## https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
def batch_normalization(inputs, is_training, epsilon = 0.001):
    # inputs should have the shape of (batch_size, width, length, channels)
    shape = [int(ii) for ii in list(inputs.get_shape()[1:])]
    scale = tf.Variable(tf.ones(shape))
    beta = tf.Variable(tf.zeros(shape))
    pop_mean = tf.Variable(tf.zeros(shape), trainable=False)
    pop_var = tf.Variable(tf.ones(shape), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * (1-epsilon) + batch_mean * epsilon)
        train_var = tf.assign(pop_var,
                              pop_var * (1-epsilon) + batch_var * epsilon)
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)