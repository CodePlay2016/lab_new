#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:03:10 2017

@author: codeplay2017
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#from __future__ import unicode_literals
#
#from builtins import open
#from builtins import str
## etc., as needed
#
#from future import standard_library
#standard_library.install_aliases()

import numpy as np
import pickle, glob, os
from PIL import Image

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
            is_epoch_over = True
        else:
            is_epoch_over = False
        start = self._index_in_epoch
        end = start + batchsize
        if shuffle:
            self.shuffle()
        self._index_in_epoch += batchsize
        return self.images[range(start, end)], self.labels[range(start, end)], is_epoch_over
    
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
        
#data_num = 8
step = 2400 # base is 20

#filepath = "/home/codeplay2017/code/lab/code/paper/realwork/image/trainset"+str(data_num)+"/test/"
#filepath = "/home/codeplay2017/code/lab/code/paper/realwork/image/image_compare/frequency_series_5speeds/test/"
filepath = "/home/codeplay2017/code/lab/code/paper/realwork/image/trainset9_30-0/compare_cwt/"
#targetpath = "../resources/py2/data4trainset"+str(data_num)+"/"
#targetpath = "../resources/py2/data4frequency_series_5speeds/"
targetpath = "../resources/py3/step"+str(step)+",30-0/"
if not os.path.exists(targetpath):
        os.system('mkdir '+targetpath)

def pickle_it(dataSet, file_name):
    with open(file_name, 'wb') as pickle_file:
        print('pickling' + file_name)
        pickle.dump(dataSet, pickle_file)
        print('over')

def main(filepath):
    '''
        transfer thousands of images into lists
        and pickle them, with lable
    '''
    
    count = 0
    num_piece = 1
    dataSet = ImgDataSet();
    file_list = glob.glob(filepath + '*.png')
    file_list_len = len(file_list)
    total = 0
    
    for infile in file_list:
        
        img = Image.open(infile)
        width, length = img.size
        img_array1 = np.array(img).reshape([width*length])
        img_array2 = np.array(img.transpose(Image.FLIP_LEFT_RIGHT)).reshape([width*length])
        img_array3 = np.array(img.transpose(Image.FLIP_TOP_BOTTOM)).reshape([width*length])
        del img
        
        # specify the type('normal,pgmt,pgsw')
        filename = infile.split('/')[-1]
        file_num = int(filename.split('_')[-1].split('.')[0])
        img_type = filename.split(',')[0]
        switcher = {
                "normal": [1,0,0],
                "pgmt":   [0,1,0],
                "pgsw":   [0,0,1]
                }
        if img_type in switcher.keys():
            img_type = switcher.get(img_type)
        else:
            print('unknown type name')
            
        # condition 1: separate the train and test data
        # condition 2: if the step doesn't meet requirement, next
        if file_num*20%step: # for train > 2000, for test < 2300/2400
            continue
        else:
            # specify whether this data is for test
            count += 1
            
            dataSet.add_data(img_array1, img_type)
#            dataSet.add_data(img_array2, img_type)
#            dataSet.add_data(img_array3, img_type)
            
            # seperate the whole data into several pieces
            # which will be joined in the further main function
            if count >= 1000 or (num_piece-1)*1000+count == 3*3000/(step/20):
                dataSet.make(shuffle=False,clean=True)
                print('images shape is ' + str(dataSet.images.shape))
    #            filename = targetpath + 'input_data'+str(data_num)+'_t_' + str(num_piece) + '.pkl'
                filename = targetpath + 'input_data_30-0_' + str(num_piece) + '.pkl'
                pickle_it(dataSet, filename)
                dataSet = ImgDataSet()
                count = 0
                num_piece += 1
        total = (num_piece-1)*1000+count
            
    print('all data picked, '+str(num_piece-1)+' data pieces')  
    print('contains '+str(total)+' data')
        
    # make all data type to numpy array
#    imgDataSet.make_dataset(shuffle=True)
#    imgDataSet.seperate_data(sep=0.9)
    
if __name__ == "__main__":
    main(filepath)
    
