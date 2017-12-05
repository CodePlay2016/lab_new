#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 10:31:06 2017

@author: codeplay2017
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#from __future__ import unicode_literals
#
#from builtins import str
## etc., as needed
#
#from future import standard_library
#standard_library.install_aliases()

import tensorflow as tf
import numpy as np
import os, glob, pickle
import matplotlib.pyplot as plt
from PIL import Image
from make_data_pai import ImgDataSet
#import matplotlib as mp

FLAGS = None

def deepnn(x, is_training):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
        x: an input tensor with the dimensions (N_examples, 240*200), where 300*200
        is the number of pixels in the image.
    Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 3), with values
        equal to the logits of classifying the digit into one of 10 classes (the
            digits 0-9). keep_prob is a scalar placeholder for the probability of
            dropout.
    """
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 1024,1, 1])

    with tf.name_scope('conv1'):
        num_feature1 = 20
        W_conv1 = weight_variable([10, 1, 1, num_feature1])
        b_conv1 = bias_variable([num_feature1])
        BN_conv1 = batch_normalization(
                tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1],
                             padding='SAME') + b_conv1, is_training=is_training)
        h_conv1 = tf.nn.relu(BN_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool(h_conv1)

    with tf.name_scope('conv2'):
        num_feature2 = 40
        W_conv2 = weight_variable([5, 1, num_feature1, num_feature2])
        b_conv2 = bias_variable([num_feature2])
        BN_conv2 = batch_normalization(
                conv2d(h_pool1, W_conv2) + b_conv2, is_training=is_training)
        h_conv2 = tf.nn.relu(BN_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool(h_conv2)
        
    with tf.name_scope('conv3'):
        num_feature3 = 40
        W_conv3 = weight_variable([5, 1, num_feature2, num_feature3])
        b_conv3 = bias_variable([num_feature3])
        BN_conv3 = batch_normalization(
                conv2d(h_pool2, W_conv3) + b_conv3, is_training=is_training)
        h_conv3 = tf.nn.relu(BN_conv3)

    with tf.name_scope('pool3'):
        h_pool3 = max_pool(h_conv3)
        
    with tf.name_scope('fc1'):
        out_size = 2048
        W_fc1 = weight_variable([128 * num_feature3, out_size])
        b_fc1 = bias_variable([out_size])
    
        h_pool2_flat = tf.reshape(h_pool3, [-1, 128*num_feature3])
        BN_fc1 = batch_normalization(
                tf.matmul(h_pool2_flat, W_fc1) + b_fc1, is_training=is_training)
        h_fc1 = tf.nn.relu(BN_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 256 features to 3 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([out_size, 3])
        b_fc2 = bias_variable([3])
    
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # notice this "SAME" param makes the conved image size the same as the original

def max_pool(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                          strides=[1, 2, 2, 1], padding='SAME') 

def weight_variable(shape, name=None):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def deconv(x, W)

# if wanna change version of this function, see utils/batch_normalizations.py
def batch_normalization(inputs, is_training, epsilon = 0.001, mode="extractable"):
    # inputs should have the shape of (batch_size, width, length, channels)
    shape = [int(ii) for ii in list(inputs.get_shape()[1:])]
    mean, var = tf.nn.moments(inputs, axes=[0])
    scale = tf.Variable(tf.ones(shape))
    beta = tf.Variable(tf.zeros(shape))
    pop_mean = tf.Variable(tf.zeros(shape), trainable=False)
    pop_var = tf.Variable(tf.ones(shape), trainable=False)
    batch_mean, batch_var = tf.nn.moments(inputs,[0])

    if mode == "extractable":
        mean, var = tf.cond(is_training, 
            lambda: (
                tf.assign(pop_mean, pop_mean * (1-epsilon) + batch_mean * epsilon),
                tf.assign(pop_var, pop_var * (1-epsilon) + batch_var * epsilon)),
            lambda: (pop_mean, pop_var))
        
    return tf.nn.batch_normalization(inputs, mean, var, beta, scale, epsilon)

source_dir = '/home/codeplay2017/code/lab/code/paper/realwork/python/observation/171108/step2400_30-0/original/'
out_dir = '/home/codeplay2017/code/lab/code/paper/realwork/python/'
model_path = os.path.join(out_dir, 'observation/171122/fft_1024_5speeds_step1/exp1/model.ckpt')

def main(): # _ means the last param
  # obersavation test  
#  time_info = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
    msg = ''
    tmsg = 'this is to extract the layer images'
    print(tmsg)
    msg += tmsg
  
    # Import data
    
    # Create the model
    x = tf.placeholder(tf.float32, [None, 1024])
    y_ = tf.placeholder(tf.float32, [None, 3])
    is_training = tf.placeholder(tf.bool)
    
    y_conv, keep_prob = deepnn(x, is_training) # result_show:[h_conv1, h_pool1, h_conv2, h_pool2] 
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
#    f_list = glob.glob(source_dir+'*.png')
    
    with open(out_dir+'resources/py3/data4fft_5speeds_1024_step1/input_data_t.pkl', 'rb') as f:
        test_set = pickle.load(f)
    datasize = test_set.num_examples()   
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        correct = 0
        for ii in range(datasize):
            img,label,_ = test_set.next_batch(1)
            valid_accuracy = accuracy.eval(feed_dict={
                        x: img, y_: label, keep_prob: 1.0, is_training: False})
#            print('predict ',tf.argmax(y_conv,1).eval(feed_dict={
#                        x: img, y_: label, keep_prob: 1.0}))
            if valid_accuracy:
                correct += 1
            
    print(correct/datasize)
            
            
#------custom functions----------------------------------
    

    
    
if __name__ == '__main__':
    main()
    
