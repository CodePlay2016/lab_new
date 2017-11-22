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

def deepnn(x):
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
        x_image = tf.reshape(x, [-1, 200, 240, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    # The third dimension of W is number of input channels, the last is the 
    # number of output channels
    with tf.name_scope('conv1'):
        num_feature1 = 3
        W_conv1 = weight_variable([5, 5, 1, num_feature1])
        b_conv1 = bias_variable([num_feature1])
        BN_conv1 = batch_normalization(
                conv2d(x_image, W_conv1) + b_conv1,is_training=False)
        h_conv1 = tf.nn.relu(BN_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        num_feature2 = 3
        W_conv2 = weight_variable([5, 5, num_feature1, num_feature2])
        b_conv2 = bias_variable([num_feature2])
        BN_conv2 = batch_normalization(
                conv2d(h_pool1, W_conv2) + b_conv2, is_training=False)
        h_conv2 = tf.nn.relu(BN_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)
        
    # Third convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv3'):
        num_feature3 = 3
        W_conv3 = weight_variable([5, 5, num_feature2, num_feature3])
        b_conv3 = bias_variable([num_feature3])
        BN_conv3 = batch_normalization(
                conv2d(h_pool2, W_conv3) + b_conv3, is_training=False)
        h_conv3 = tf.nn.relu(BN_conv3)

    # Fully connected layer 1 -- after 2 round of downsampling, our 300x200 image
    # is down to 75x50x64 feature maps -- maps this to 256 features.
    with tf.name_scope('fc1'):
        out_size = 4096
        W_fc1 = weight_variable([50 * 60 * num_feature3, out_size])
        b_fc1 = bias_variable([out_size])
    
        h_pool2_flat = tf.reshape(h_conv3, [-1, 50*60*num_feature3])
        BN_fc1 = batch_normalization(
                tf.matmul(h_pool2_flat, W_fc1) + b_fc1,is_training=False)
        h_fc1 = tf.nn.relu(BN_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 256 features to 3 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 3])
        b_fc2 = bias_variable([3])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  
    return y_conv, keep_prob, [h_conv1, h_conv2, h_conv3, h_fc1]   


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # notice this "SAME" param makes the conved image size the same as the original

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME') 

def weight_variable(shape, name=None):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def batch_normalization(inputs, is_training, epsilon=0.001, escape=False):
    # inputs should have the shape of (batch_size, width, length, channels)
    shape = [int(ii) for ii in list(inputs.get_shape()[1:])]
    scale = tf.Variable(tf.ones(shape))
    beta = tf.Variable(tf.zeros(shape))
    pop_mean = tf.Variable(tf.zeros(shape), trainable=False)
    pop_var = tf.Variable(tf.ones(shape), trainable=False)
    if escape:
        return inputs
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

source_dir = '/home/codeplay2017/code/lab/code/paper/realwork/python/observation/171108/step2400_30-0/original/'
out_dir = '/home/codeplay2017/code/lab/code/paper/realwork/python/'
model_path = os.path.join(out_dir, 'observation/171108/step2400_30-0/model_no_mirror_bn0.1/model.ckpt')

def main(): # _ means the last param
  # obersavation test  
#  time_info = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
    msg = ''
    tmsg = 'this is to extract the layer images'
    print(tmsg)
    msg += tmsg
  
    # Import data
    
    # Create the model
    x = tf.placeholder(tf.float32, [None, 200*240])
    y_ = tf.placeholder(tf.float32, [None, 3])
    y_conv, keep_prob, result_show = deepnn(x) # result_show:[h_conv1, h_pool1, h_conv2, h_pool2] 
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
#    f_list = glob.glob(source_dir+'*.png')
    
    with open(out_dir+'resources/py3/step2400,30-0/input_data_30-0_1.pkl', 'rb') as f:
        test_set = pickle.load(f)
    datasize = test_set.num_examples()   
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        correct = 0
        for ii in range(datasize):
            img,label,_ = test_set.next_batch(2)
            valid_accuracy = accuracy.eval(feed_dict={
                        x: img, y_: label, keep_prob: 1.0})
#            print('predict ',tf.argmax(y_conv,1).eval(feed_dict={
#                        x: img, y_: label, keep_prob: 1.0}))
            if valid_accuracy:
                correct += 1
            
    print(correct/datasize)
            
            
#            img_file_name = file.split('/')[-1]
#            img, img_array, img_type, img_type_vector, save_dir = load_img2array(
#                    source_dir, img_file_name)
        
#            predict = sess.run(tf.argmax(y_conv,1), feed_dict={
#                x: img_array, keep_prob: 1.0})
#            print(predict,' '+img_type)
#            conv1_array = sess.run(result_show[0], feed_dict={
#                x: img_array, keep_prob: 1.0})
#            conv2_array = sess.run(result_show[1], feed_dict={
#                x: img_array, keep_prob: 1.0})
#            conv1_array = conv1_array[0,:,:,:]
#            conv2_array = conv2_array[0,:,:,:]
#            if predict==np.argmax(img_type_vector):
#                correct+=1
#        print(correct/len(f_list))
            
            
#------custom functions----------------------------------
    
def show(array, cmap='gray'):
    return plt.imshow(array, origin='lower', cmap=cmap)
        
def plot_img2curve(img_array, num_line=10, normalize=True):
    _, length = img_array.shape
    img_array = img_array[0:num_line,:]
    if normalize:
        img_array = img_normalize(img_array)
    plt.plot(img_array.reshape(num_line*length,1), linewidth=1)
    

def save_image_fromarray(array, savepath):
    if not os.path.exists(savepath):
        os.system('mkdir '+savepath)
    num_channel = array.shape[-1]
    width = array.shape[1]
    length = array.shape[2]
    rgb_img_array = np.zeros([width, length, 3], dtype=np.uint8)
    for ii in range(num_channel):
    	 # the array should be rescaled into 0~255 uint8 type
        img_array = img_normalize(array[0,:,:,ii])
        img = Image.fromarray(img_array)
        img.save(savepath+str(ii)+'.png')
        rgb_img_array[:,:,ii] = img_array
    Image.fromarray(rgb_img_array, mode="RGB").save(savepath+"_rgb.png")
    
        
def img_normalize(img_array, normalize=True, reverse=False, polarize=False):
    minVal = np.amin(img_array)
    maxVal = np.amax(img_array)
    if normalize:
        result = np.uint8((img_array-minVal)/(maxVal-minVal)*255)
    else:
        result = np.uint8(img_array*255)
    if reverse:
        result = 255-result
    if polarize:
        result[result>0]=255
    return result

def print_predict(index):
    if index == 0:
        estimated_msg = 'estimated: this data comes from normal gearbox'
    elif index == 1:
        estimated_msg = 'estimated: this data comes from gearbox with planetary gear missing tooth'
    elif index == 2:
        estimated_msg = 'estimated: this data comes from gearbox with planetary gear surface worn'
    else:
        estimated_msg = 'index error!!'
    return estimated_msg

def load_img2array(source_dir, img_file_name):
    img = Image.open(source_dir + img_file_name)
    width, length = img.size
    img_array = np.array(img).reshape([1, width*length])
  
    img_number = img_file_name.split('_')[1]
    img_speed = img_file_name.split('-')[1].split(',')[0]
    img_type = img_file_name.split(',')[0]
  
    save_file_name_template = img_type+'_'+img_speed+'_'+img_number
    save_dir = os.path.join(out_dir,'observation/fft/'+save_file_name_template+'/') # the directory for saving images
    os.system('mkdir '+save_dir)
    os.system('cp '+source_dir+img_file_name+' '+save_dir)
    switcher = {
        "normal": [1,0,0],
        "pgmt":   [0,1,0],
        "pgsw":   [0,0,1]
        }
    img_type_vector = np.array(switcher.get(img_type)).reshape([1,3])
    return img, img_array, img_type, img_type_vector, save_dir
    
    
if __name__ == '__main__':
    main()
    
