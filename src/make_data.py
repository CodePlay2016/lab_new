#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:03:10 2017

@author: codeplay2017
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open
# etc., as needed

from future import standard_library
standard_library.install_aliases()

import numpy as np
import pickle, glob
from PIL import Image

class ImgDataSet():
    def __init__(self, imagelist=[], labellist=[]):
        self._imagelist = imagelist
        self._labellist = labellist
        self._index_in_epoch = 0

    def add_data(self, img, label):
        self._imagelist.append(img)
        self._labellist.append(label)

    def make_dataset(self, shuffle=False, sep=0.5): 
        #make the list to numpy.array
        #do this after all data are added
        #if shuffle is needed, this method will do the shuffle to both lists and
        #arrays
            
        self.images = np.array(self._imagelist)
        self.labels = np.array(self._labellist)
        if shuffle:
            self.shuffle()
    
    def shuffle(self):
        index = list(range(self.num_examples()))
        np.random.shuffle(index)
        self._images = self.images[index] # randomly arranged
        self._labels = self.labels[index] # randomly arranged
#        return self._images, self._labels

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
        return self._images[range(start, end)], self._labels[range(start, end)]
    
    def seperate_data(self, sep=0.5):
        num = self.num_examples()
        train_num = int(num * sep)
        self.train = ImgDataSet(list(self._images[:train_num]), list(self._labels[:train_num]))
        self.test = ImgDataSet(list(self._images[train_num:]), list(self._labels[train_num:]))
        self.train.make_dataset(True)
        self.test.make_dataset(True)

filepath = "/home/codeplay2017/code/lab/code/paper/realwork/image/trainset1/"
targetpath = "../resources/"

def pickle_img_data(filepath, pool=False):
    '''
        transfer thousands of images into lists
        and pickle them, with lable
    '''
    imgDataSet = ImgDataSet();
    for infile in glob.glob(filepath + "*.png"):
        img = Image.open(infile)
        width, length = img.size

        img_array = np.array(img)
        if pool:
            pass
        
        # reshape the image to an 1-d data
        img_array.reshape([width*length])
        
        # specify the type('normal,pgmt')
        img_type = infile.split(sep="/")[-1].split(sep=",")[0]
        switcher = {
                "normal": [1,0,0],
                "pgmt":   [0,1,0],
                "pgsw":   [0,0,1]
                }
        if img_type in switcher.keys():
            img_type = switcher.get(img_type)
        else:
            print('unknown type name')
        imgDataSet.add_data(img_array, img_type)
        
    # make all data type to numpy array
    imgDataSet.make_dataset(shuffle=True)
    imgDataSet.seperate_data(sep=0.9)
    
    # pickle the data
    with open(targetpath + 'input_data1_p2.pkl', 'wb') as pickle_file:
        pickle.dump(imgDataSet, pickle_file, protocol=2)
    
if __name__ == "__main__":
    pickle_img_data(filepath)
    
        
        
        
        
        
