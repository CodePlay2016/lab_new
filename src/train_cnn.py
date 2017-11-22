#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 09:47:10 2017

@author: codeplay2017
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#from builtins import str
## etc., as needed
#
##from future import standard_library
#standard_library.install_aliases()

import tensorflow as tf
import numpy as np
import tempfile
import pickle, time, os, argparse

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

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  # The third dimension of W is number of input channels, the last is the 
  # number of output channels
  with tf.name_scope('conv1'):
    num_feature1 = 20
    W_conv1 = weight_variable([5, 5, 1, num_feature1])
    b_conv1 = bias_variable([num_feature1])
    h_conv1 = tf.nn.relu(batch_normalization(conv2d(x_image, W_conv1) + b_conv1, is_training=True))
#    layer_image1 = tf.transpose(h_conv1, perm=[3,1,2,0])
#    tf.summary.image('conv2',
#                     layer_image1[:,:,:,-1].reshape([num_feature1,200,240,1]),
#                     max_outputs=num_feature1/2)
  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)
#    tf.summary.image('pool1', h_pool1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    num_feature2 = 40
    W_conv2 = weight_variable([5, 5, num_feature1, num_feature2])
    b_conv2 = bias_variable([num_feature2])
    h_conv2 = tf.nn.relu(batch_normalization(conv2d(h_pool1, W_conv2) + b_conv2, is_training=True))
#    layer_image2 = tf.transpose(h_conv2, perm=[3,1,2,0])
#    tf.summary.image('conv2', layer_image2[:,:,:,-1], max_outputs=num_feature2/2)
#    tf.summary.image('conv2', h_conv2[:,:,:,np.random.randint(0,num_feature1)])

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)
#    tf.summary.image('pool2', h_pool2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 300x200 image
  # is down to 75x50x64 feature maps -- maps this to 256 features.
  with tf.name_scope('fc1'):
    out_size = 1024
    W_fc1 = weight_variable([50 * 60 * num_feature2, out_size])
    b_fc1 = bias_variable([out_size])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 50*60*num_feature2])
    h_fc1 = tf.nn.relu(batch_normalization(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, is_training=True))

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

# this is a simpler version of Tensorflow's 'official' version. See:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
def batch_normalization(inputs, is_training, epsilon = 0.001):
    # inputs should have the shape of (batch_size, width, length, channels)
    shape = [int(ii) for ii in list(inputs.get_shape())]
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

    
def main(): # _ means the last param
  # Import data
  data_num = str(7)
  data_dir = '/home/codeplay2017/code/lab/code/paper/realwork/python/resources/py2/data4trainset1'
  print('this is for trainset8, multi-speed, ')
  
  trainset = ImgDataSet()
  testset = ImgDataSet()
  data_num = str(8)
  num_trainfile = 11
  num_testfile = 1
  for ii in range(num_trainfile):
    data_path = os.path.join(data_dir,'input_data'+data_num+'_'+str(ii+1)+'.pkl')
    with tf.gfile.GFile(data_path, 'rb') as f:
      data = pickle.load(f)
    trainset.join_data(data)
  for ii in range(num_testfile):
    data_path = os.path.join(data_dir,'input_data'+data_num+'_t_'+str(ii+1)+'.pkl')
    with tf.gfile.GFile(data_path, 'rb') as f:
      data = pickle.load(f)
    testset.join_data(data)
  testset.make(shuffle=True,clean=True)
  trainset.make(shuffle=True,clean=True)
  
#  dataset.seperate_data(sep=0.9)
  
  # Create the model
  x = tf.placeholder(tf.float32, [None, 200*240])
  y_ = tf.placeholder(tf.float32, [None, 3])

  y_conv, keep_prob = deepnn(x)

  # Define loss and optimizer
  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss-cross_entropy', cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    time_info = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
    print(time_info)
    print('this is for trainset8 without batch normalization')
    output_dir = FLAGS.checkpointDir+time_info+'/'
    model_path = os.path.join(output_dir, 'model.ckpt')
#    img_path = os.path.join(output_dir, 'layer_image')
    summary_path = os.path.join(output_dir, 'train_summary') 
    print('this experiment is for trainset'+data_num)
    print('model is saved to:'+model_path)
    print('summary is saved to:'+summary_path)
    
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir=summary_path)
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    
    valid_accuracy_list=[]
    curve_list = [[],[],[]]
    for i in range(20000):
      train_batch = trainset.next_batch(200)
      valid_batch = testset.next_batch(200)
      if i and i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: train_batch[0], y_: train_batch[1], keep_prob: 1.0})
        valid_accuracy = accuracy.eval(feed_dict={
            x: valid_batch[0], y_: valid_batch[1], keep_prob: 1.0})
        
        summary,_ = sess.run([merged_summary, train_step], feed_dict={
            x: valid_batch[0], y_: valid_batch[1], keep_prob: 1.0})
        summary_writer.add_summary(summary, i)
        
        valid_accuracy_list.append(valid_accuracy)
        valid_accuracy_average = sum(valid_accuracy_list)/len(valid_accuracy_list) if len(valid_accuracy_list)<10 else sum(valid_accuracy_list[-10:])/10
        msg = 'step |%d|, train accuracy |%.2g|, valid |%.3g|, average for last 10 valid |%.4g|' % (
                i, train_accuracy, valid_accuracy, valid_accuracy_average)
        
        curve_list[0].append(train_accuracy)
        curve_list[1].append(valid_accuracy)
        curve_list[2].append(valid_accuracy_average)
        print(msg)
      train_step.run(feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob: 0.5})
    
    # save the trained model
    saver.save(sess=sess, save_path=model_path)
#    conv1_img_path = os.path.join(img_path, 'con1_img')
#    conv1_img_array = sess.run(h_conv1)
#    save_image_fromarray(conv1_img_array[], conv1_img_path)

    test_batch = testset.next_batch(200)
    test_accuracy = 'test accuracy %.2g' % accuracy.eval(feed_dict={
        x: test_batch[0], y_: test_batch[1], keep_prob: 1.0})
    print(test_accuracy)

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
    main()
