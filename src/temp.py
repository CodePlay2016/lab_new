#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:44:41 2017

@author: codeplay2017
"""

import glob, os

file_path = "/home/codeplay2017/code/lab/code/paper/realwork/image/trainset9_30-0/compare_cwt/"
save_path = "/home/codeplay2017/code/lab/code/paper/realwork/python/observation/171108/step2400_30-0/original/"
file_list = glob.glob(file_path+'*.png')
step = 2400

for file in file_list:
    file_name = file.split('/')[-1]
    number = int(file.split('_')[-1].split('.')[0])
    if not number*20 % step:
        os.system('cp '+file+' '+save_path+file_name)
        
    