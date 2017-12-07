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
import scipy.io as sio
import os, glob, pickle
import matplotlib.pyplot as plt
from PIL import Image
import make_data_pai as md
from tensorflow.python import pywrap_tensorflow
#import matplotlib as mp

FLAGS = None

def deepnn(x, is_training, keep_prob):
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
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 2048,1, 1])

    # First convolutional layer - maps one grayscale image to 5 feature maps.
    # The third dimension of W is number of input channels, the last is the 
    # number of output channels
    with tf.name_scope('conv1'):
        num_feature1 = 20
        W_conv1 = weight_variable([64, 1, 1, num_feature1], name='W_conv1')
        b_conv1 = bias_variable([num_feature1], name='b_conv1')
        BN_conv1 = batch_normalization(
                tf.nn.conv2d(x_image, W_conv1, strides=[1, 16, 1, 1],
                             padding='SAME') + b_conv1, is_training=is_training)[0]
        h_conv1 = tf.nn.relu(BN_conv1, name='h_conv1')

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool(h_conv1)

    # Second convolutional layer -- maps 5 feature maps to 5
    with tf.name_scope('conv2'):
        num_feature2 = 40
        W_conv2 = weight_variable([5, 1, num_feature1, num_feature2], name='W_conv2')
        b_conv2 = bias_variable([num_feature2], name='b_conv2')
        BN_conv2 = batch_normalization(
                conv2d(h_pool1, W_conv2) + b_conv2, is_training=is_training)[0]
        h_conv2 = tf.nn.relu(BN_conv2, name='h_conv2')

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool(h_conv2)
        
    with tf.name_scope('conv3'):
        num_feature3 = 40
        W_conv3 = weight_variable([5, 1, num_feature2, num_feature3], name='W_conv3')
        b_conv3 = bias_variable([num_feature3], name='b_conv3')
        BN_conv3 = batch_normalization(
                conv2d(h_pool2, W_conv3) + b_conv3, is_training=is_training)[0]
        h_conv3 = tf.nn.relu(BN_conv3, name='h_conv3')
        
    with tf.name_scope('pool3'):
        h_pool3 = max_pool(h_conv3)
        
    with tf.name_scope('conv4'):
        num_feature4 = 40
        W_conv4 = weight_variable([5, 1, num_feature3, num_feature4], 'W_conv4')
        b_conv4 = bias_variable([num_feature4], 'W_conv4')
        BN_conv4 = batch_normalization(
                conv2d(h_pool3, W_conv4) + b_conv4, is_training=is_training)[0]
        h_conv4 = tf.nn.relu(BN_conv4, 'h_conv4')
        
    # Second pooling layer.
    with tf.name_scope('pool4'):
        h_pool4 = max_pool(h_conv4)
        
    # Fully connected layer 1 -- after 2 round of downsampling, our 300x200 image
    # is down to 75x50x64 feature maps -- maps this to 256 features.
    with tf.name_scope('fc1'):
        out_size = 2048
        W_fc1 = weight_variable([8 * num_feature4, out_size], 'W_fc1')
        b_fc1 = bias_variable([out_size], 'b_fc1')
    
        h_pool2_flat = tf.reshape(h_pool4, [-1, 8*num_feature4])
        BN_fc1,pop_mean = batch_normalization(
                tf.matmul(h_pool2_flat, W_fc1) + b_fc1, is_training=is_training)
        h_fc1 = tf.nn.relu(BN_fc1, 'h_fc1')

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 256 features to 3 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([out_size, 3], 'W_fc2')
        b_fc2 = bias_variable([3], 'b_fc2')
    
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, pop_mean


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # notice this "SAME" param makes the conved image size the same as the original

def max_pool(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                          strides=[1, 2, 2, 1], padding='SAME') 

def weight_variable(initial, name=None):
    """weight_variable generates a weight variable of a given shape."""
    return tf.Variable(initial=initial, name=name)

def bias_variable(initial, name=None):
    """bias_variable generates a bias variable of a given shape."""
    return tf.Variable(initial, name=name)

# if wanna change version of this function, see utils/batch_normalizations.py
def batch_normalization(inputs, is_training, epsilon = 0.001, mode="extractable",
                        pop_mean=None, pop_var=None, beta=None, scale=None):
    # inputs should have the shape of (batch_size, width, length, channels)
#    out_size = inputs.shape[-1]
    shape = [int(ii) for ii in list(inputs.get_shape()[1:])]
    scale = tf.Variable(tf.ones(shape), name='BN_scale')
    beta = tf.Variable(tf.zeros(shape), name='BN_beta')
    pop_mean = tf.Variable(tf.zeros(shape), trainable=False, name='BN_pop_mean')
    pop_var = tf.Variable(tf.ones(shape), trainable=False, name='BN_pop_var')
    batch_mean, batch_var = tf.nn.moments(inputs,[0])
    if mode is "extractable":
        pop_mean, pop_var = tf.cond(is_training, 
            lambda: (
                tf.assign(pop_mean, pop_mean * (1-epsilon) + batch_mean * epsilon),
                tf.assign(pop_var, pop_var * (1-epsilon) + batch_var * epsilon)),
            lambda: (pop_mean, pop_var))
    elif mode is "adaptive":
        tf.assign(pop_mean, batch_mean)
        tf.assign(pop_var, batch_var)
        
    return (tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon),
            pop_mean)

source_dir = '/home/codeplay2017/code/lab/code/paper/realwork/image/wen_data/raw_divided/time_series_step1_2048_5speeds/'
out_dir = '/home/codeplay2017/code/lab/code/paper/realwork/python/'
model_path = os.path.join(out_dir, 'observation/171206/train1speedtestanother/2017-12-07_00:39:24/model.ckpt')

def main(): # _ means the last param
  # obersavation test  
#  time_info = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
  
    # Import data
    
    # Create the model
    x = tf.placeholder(tf.float32, [None, 2048])
    y_ = tf.placeholder(tf.float32, [None, 3])
    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)
    y_conv, pop_mean = deepnn(x, is_training, keep_prob)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    ### load data
    flist = glob.glob(source_dir+'*-50,*.mat')
    ### begin test
    
    with tf.Session() as sess:
        #1.load raw data
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        correct = 0
        count = 0
        for file in flist:
            (matdata, data_type, source_type,
             speed, num_of_data, length) = md.prepare_data(file, fft=False, mirror=False)
        #2.test(raw data)
            testsize = 50000
#            for ii in range(testsize):
            count += 1
            test = matdata[40000:testsize,:]
            length = test.shape[0]
#                test_accuracy = accuracy.eval(feed_dict={
#                        x: test.reshape(1,2048), y_: np.array(data_type).reshape(1,3),
#                            keep_prob: 1.0, is_training: False})
            mean = sess.run(pop_mean,feed_dict={
                    x: test, y_: np.array([data_type]*length),
                    keep_prob: 1.0, is_training: False})
            test_accuracy = accuracy.eval(feed_dict={
                    x: test, y_: np.array([data_type]*length),
                    keep_prob: 1.0, is_training: False})
    #                y_out = y_conv.eval(feed_dict={
    #                        x: test.reshape(1,2048), y_: np.array(data_type).reshape(1,3),
    #                        keep_prob: 1.0, is_training: False})
            correct += test_accuracy
#                if not test_accuracy:
#                    print(str(np.array(data_type).reshape(1,3))+'--'+str(y_out)+'-Error')
#                else:
#                    print(str(np.array(data_type).reshape(1,3))+'--'+str(y_out))
            print(mean)
                
        #3. 
        print(correct/count)
            


            
#########------custom functions----------------------------------
def predict(data, accuracy_tensor, model_path):
    datasize = data.num_examples()   
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        correct = 0
        for ii in range(datasize):
            img,label,_ = data.next_batch(1)
            valid_accuracy = accuracy_tensor.eval(feed_dict={
                        x: img, y_: label, keep_prob: 1.0, is_training: False})
#            print('predict ',tf.argmax(y_conv,1).eval(feed_dict={
#                        x: img, y_: label, keep_prob: 1.0}))
            if valid_accuracy:
                correct += 1
            
    print(correct/datasize)
    
def load(path, ftype='mat'):
    if ftype == 'mat':
        data = sio.loadmat(path)['originSet']
    elif ftype == 'pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
    return data
        

    
    
if __name__ == '__main__':
    main()
    # Read data from checkpoint file
    reader = pywrap_tensorflow.NewCheckpointReader(model_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # Print tensor name and values
    keylist = []
    for key in var_to_shape_map:
        keylist.append(key)
    list.sort(keylist)
#    wanted_tensor = reader.get_tensor('conv3/BN_pop_var')