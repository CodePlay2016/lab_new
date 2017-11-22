#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 10:31:06 2017

@author: codeplay2017
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import str
# etc., as needed

from future import standard_library
standard_library.install_aliases()

import tensorflow as tf
import numpy as np
import pickle, os, argparse

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
    # show the image in tensorboard.
    # third argument is max shown num
#    tf.summary.image('input', x_image, 10) 

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  # The third dimension of W is number of input channels, the last is the 
  # number of output channels
  with tf.name_scope('conv1'):
    num_feature1 = 20
    W_conv1 = weight_variable([5, 5, 1, num_feature1])
    b_conv1 = bias_variable([num_feature1])
    h_conv1 = tf.nn.relu(batch_normalization(conv2d(x_image, W_conv1) + b_conv1
                                             , out_size=num_feature1))
  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    num_feature2 = 40
    W_conv2 = weight_variable([5, 5, num_feature1, num_feature2])
    b_conv2 = bias_variable([num_feature2])
    h_conv2 = tf.nn.relu(batch_normalization(conv2d(h_pool1, W_conv2) + b_conv2
                                             , out_size=num_feature2))
  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 300x200 image
  # is down to 75x50x64 feature maps -- maps this to 256 features.
  with tf.name_scope('fc1'):
    out_size = 1024
    W_fc1 = weight_variable([50 * 60 * num_feature2, out_size])
    b_fc1 = bias_variable([out_size])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 50*60*num_feature2])
    h_fc1 = tf.nn.relu(batch_normalization(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, out_size))

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
  return y_conv, keep_prob


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

def batch_normalization(data, out_size, axes=[0], epsilon=0.001):
    # do the batch normalization
        # calculate the mean and variance of 'conv_Wx_Plus_b1'
        # axes=[0] means it calculate in the dimension of batch_size
    mean, var = tf.nn.moments(data, axes=[0])
        # the two params below will be auto-adjusted by tensorflow
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    return tf.nn.batch_normalization(data, mean, var, shift, scale, epsilon)
def main(_): # _ means the last param
  # Import data
  dataset = ImgDataSet()
  file_num = 24
  for ii in range(file_num):
    filepath = os.path.join(FLAGS.buckets, 'input_data6_' + str(ii+1) + '.pkl')
    with tf.gfile.GFile(filepath, 'rb') as f:
      data = pickle.load(f)
    dataset.join_data(data)
  dataset.make(shuffle=True,clean=True)
  
  # Create the model
  x = tf.placeholder(tf.float32, [None, 200*240])
  y_ = tf.placeholder(tf.float32, [None, 3])

  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  with tf.Session() as sess:
    model_dir = FLAGS.checkpointDir+'/'+'2017-09-09_23:55:15/'
    model_path = os.path.join(model_dir, 'model.ckpt')
    print('restoring model from:'+model_path)
    
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    
    test_accuracy_list=[]
    for i in range(50):
      test_batch = dataset.next_batch(200)
      test_accuracy = accuracy.eval(feed_dict={
        x: test_batch[0], y_: test_batch[1], keep_prob: 1.0})
        
      test_accuracy_list.append(test_accuracy)
      test_accuracy_average = sum(test_accuracy_list)/len(test_accuracy_list) if len(test_accuracy_list)<10 else sum(test_accuracy_list[-10:])/10
      msg = 'step |%d|, test accuracy |%.2g|, average for last 10 test |%.4g|' % (
                i, test_accuracy, test_accuracy_average)
      print(msg)
    
# this is the data structure for my dataset
class ImgDataSet():
    # arguments:
    # imagelist, labelist, the first dimension must be number of examples
    # these two lists must contain elements of numpy array type
    # if shuffle == True, all the data will be arranged randomly
    def __init__(self, image_array=np.array([]), label_array=np.array([]), shuffle=False):
        self.clean()
        self.images = np.array(self._imagelist)
        self.labels = np.array(self._labellist)
        if image_array.size:
            self.images = image_array[np.arange(image_array.shape[0])]
            self.labels = label_array[np.arange(label_array.shape[0])]
        if shuffle:
            self.shuffle()
        self._index_in_epoch = 0

    def add_data(self, img, label): 
        self._imagelist.append(list(img))
        self._labellist.append(list(label))
    
    # generally, when packaging data, when join the datasets, set clean=False
    # otherwise, set clean=True    
    def make(self, shuffle=True, clean=False):
        if len(self._imagelist):
            self.images = np.array(self._imagelist)
            self.labels = np.array(self._labellist)
        else:
            self.add_data(self.images, self.labels)
        if shuffle:
            self.shuffle()
        if clean:
            self.clean()
        else:
            self._imagelist = list(self.images)
            self._labellist = list(self.labels)

    def clean(self):
        self._imagelist = []
        self._labellist = []

    def shuffle(self):
        index = list(range(self.num_examples()))
        np.random.shuffle(index)
        self.images = self.images[index] # randomly arranged
        self.labels = self.labels[index] # randomly arranged

    def num_examples(self):
        return self.images.shape[0] # return the first dimension, num of samples
    
    def next_batch(self, batchsize, shuffle=False):
        if self._index_in_epoch + batchsize >= self.num_examples():
            self._index_in_epoch = 0
            shuffle = True
        start = self._index_in_epoch
        end = start + batchsize
        if shuffle:
            self.shuffle()
        self._index_in_epoch += batchsize
        return self.images[range(start, end)], self.labels[range(start, end)]
    
    def seperate_data(self, sep=0.5):
        num = self.num_examples()
        train_num = int(num * sep)
        _tempImages = self.images
        _tempLabels = self.labels
        self.train = ImgDataSet(_tempImages[:train_num], _tempLabels[:train_num])
        self.test = ImgDataSet(_tempImages[train_num:], _tempLabels[train_num:])
        del _tempImages
        del _tempLabels
        self.train.make(shuffle=True,clean=True)
        self.test.make(shuffle=True,clean=True)
    
    def join_data(self, other):
        if self.num_examples() != 0:
            self.images = np.concatenate((self.images, other.images), axis=0)
            self.labels = np.concatenate((self.labels, other.labels), axis=0)
        else:
            self.images = other.images[np.arange(other.num_examples())]
            self.labels = other.labels[np.arange(other.num_examples())]
    
    def isEmpty(self):
        return True if self.num_examples() == 0 else False

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # path to load data
  parser.add_argument('--buckets', type=str,
                      help='input data path')
  parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main)
