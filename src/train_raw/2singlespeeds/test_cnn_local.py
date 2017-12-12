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
import make_data_pai as md
from tensorflow.python import pywrap_tensorflow
#import matplotlib as mp

####-----------------------------------------------------------------------
source_dir = '/home/codeplay2017/code/lab/code/paper/realwork/image/wen_data/raw_divided/angle_series_step1_2048_5speeds/'
out_dir = '/home/codeplay2017/code/lab/code/paper/realwork/python/'
#model_path = os.path.join(out_dir, 'observation/171206/train1speedtestanother/2017-12-06_15:48:37/model.ckpt')
model_path = os.path.join(out_dir, 'observation/171213/angle_5speeds_train1/2017-12-08_14:26:54/model.ckpt')

####-----------------------------------------------------------------------
reader = pywrap_tensorflow.NewCheckpointReader(model_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
keylist = []
for key in var_to_shape_map:
    keylist.append(key)
list.sort(keylist)
####-----------------------------------------------------------------------
####-----------------------------------------------------------------------



def deepnn(x, is_training, keep_prob):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 2048,1, 1])

    with tf.name_scope('conv1'):
        W_conv1 = tf.Variable(reader.get_tensor('conv1/W_conv1'))
        b_conv1 = bias_variable(reader.get_tensor('conv1/b_conv1'))
        BN_conv1 = batch_normalization(
                tf.nn.conv2d(x_image, W_conv1, strides=[1, 16, 1, 1],
                             padding='SAME') + b_conv1, is_training=is_training,
                             mean=reader.get_tensor('conv1/BN_pop_mean'),
                             var=reader.get_tensor('conv1/BN_pop_var'),
                             beta=reader.get_tensor('conv1/BN_beta'),
                             scale=reader.get_tensor('conv1/BN_scale'))[0]
        h_conv1 = tf.nn.relu(BN_conv1, name='h_conv1')

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool(h_conv1)

    # Second convolutional layer -- maps 5 feature maps to 5
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(reader.get_tensor('conv2/W_conv2'), name='W_conv2')
        b_conv2 = bias_variable(reader.get_tensor('conv2/b_conv2'), name='b_conv2')
        BN_conv2 = batch_normalization(
                conv2d(h_pool1, W_conv2) + b_conv2, is_training=is_training,
                             mean=reader.get_tensor('conv2/BN_pop_mean'),
                             var=reader.get_tensor('conv2/BN_pop_var'),
                             beta=reader.get_tensor('conv2/BN_beta'),
                             scale=reader.get_tensor('conv2/BN_scale'))[0]
        h_conv2 = tf.nn.relu(BN_conv2, name='h_conv2')

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool(h_conv2)
        
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(reader.get_tensor('conv3/W_conv3'))
        b_conv3 = bias_variable(reader.get_tensor('conv3/b_conv3'))
        BN_conv3 = batch_normalization(
                conv2d(h_pool2, W_conv3) + b_conv3, is_training=is_training,
                             mean=reader.get_tensor('conv3/BN_pop_mean'),
                             var=reader.get_tensor('conv3/BN_pop_var'),
                             beta=reader.get_tensor('conv3/BN_beta'),
                             scale=reader.get_tensor('conv3/BN_scale'))[0]
        h_conv3 = tf.nn.relu(BN_conv3, name='h_conv3')
        
    with tf.name_scope('pool3'):
        h_pool3 = max_pool(h_conv3)
        
    with tf.name_scope('conv4'):
        num_feature4 = 40
        W_conv4 = weight_variable(reader.get_tensor('conv4/W_conv4'))
        b_conv4 = bias_variable(reader.get_tensor('conv4/b_conv4'))
#        b_conv4 = bias_variable(reader.get_tensor('conv4/W_conv4_1'))
        BN_conv4 = batch_normalization(
                conv2d(h_pool3, W_conv4) + b_conv4, is_training=is_training,
                             mean=reader.get_tensor('conv4/BN_pop_mean'),
                             var=reader.get_tensor('conv4/BN_pop_var'),
                             beta=reader.get_tensor('conv4/BN_beta'),
                             scale=reader.get_tensor('conv4/BN_scale'))[0]
        h_conv4 = tf.nn.relu(BN_conv4, 'h_conv4')
        
    # Second pooling layer.
    with tf.name_scope('pool4'):
        h_pool4 = max_pool(h_conv4)
        
    # Fully connected layer 1 -- after 2 round of downsampling, our 300x200 image
    # is down to 75x50x64 feature maps -- maps this to 256 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable(reader.get_tensor('fc1/W_fc1'))
        b_fc1 = bias_variable(reader.get_tensor('fc1/b_fc1'))
    
        h_pool2_flat = tf.reshape(h_pool4, [-1, 8*num_feature4])
        BN_fc1,pop_mean = batch_normalization(
                tf.matmul(h_pool2_flat, W_fc1) + b_fc1, is_training=is_training,
                             mean=reader.get_tensor('fc1/BN_pop_mean'),
                             var=reader.get_tensor('fc1/BN_pop_var'),
                             beta=reader.get_tensor('fc1/BN_beta'),
                             scale=reader.get_tensor('fc1/BN_scale'))
        h_fc1 = tf.nn.relu(BN_fc1, 'h_fc1')

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 256 features to 3 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable(reader.get_tensor('fc2/W_fc2'))
        b_fc2 = bias_variable(reader.get_tensor('fc2/b_fc2'))
    
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
    return tf.Variable(initial, name=name)

def bias_variable(initial, name=None):
    """bias_variable generates a bias variable of a given shape."""
    return tf.Variable(initial, name=name)

# if wanna change version of this function, see utils/batch_normalizations.py
def batch_normalization(inputs, is_training, epsilon = 0.01, mode="adaptiv",
                        mean=None, var=None, beta=None, scale=None):
    # inputs should have the shape of (batch_size, width, length, channels)
#    out_size = inputs.shape[-1]
    batch_mean, batch_var = tf.nn.moments(inputs,[0])
    if mode is "adaptive":
        pop_mean = tf.Variable(mean)*(1-epsilon) + batch_mean*(epsilon)
        pop_var = tf.Variable(var)*(1-epsilon) + batch_var*(epsilon)
    else:
        pop_mean = tf.Variable(mean)
        pop_var  = tf.Variable(var)
    pop_scale = tf.Variable(scale)
    pop_beta = tf.Variable(beta)
    
        
    return (tf.nn.batch_normalization(inputs, pop_mean, pop_var, pop_beta, pop_scale, epsilon),
            pop_mean)



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
        sess.run(tf.global_variables_initializer())
#        saver = tf.train.Saver()
#        saver.restore(sess, model_path)
        correct = 0
        count = 0
        for file in flist:
            (matdata, data_type, source_type,
             speed, num_of_data, length) = md.prepare_data(file, fft=False, mirror=False)
        #2.test(raw data)
            testsize = 50000
#            for ii in range(testsize):
            count += 1
            matdata = matdata[45000:testsize,:]
            length = matdata.shape[0]
#                test_accuracy = accuracy.eval(feed_dict={
#                        x: test.reshape(1,2048), y_: np.array(data_type).reshape(1,3),
#                            keep_prob: 1.0, is_training: False})
            mean = sess.run(pop_mean,feed_dict={
                    x: matdata, y_: np.array([data_type]*length),
                    keep_prob: 1.0, is_training: False})
            test_accuracy = accuracy.eval(feed_dict={
                    x: matdata, y_: np.array([data_type]*length),
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
    
def load(path, ftype='mat'):
    if ftype == 'mat':
        data = sio.loadmat(path)['originSet']
    elif ftype == 'pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
    return data
        
    
    
if __name__ == '__main__':
    main()
