#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:23:08 2017

@author: codeplay2017
"""
import pickle

filepath = '/home/codeplay2017/code/lab/code/paper/realwork/python/resources/models/model_2017-09-27_19:40:40/'

curvelist = [[],[],[]]
with open(filepath+'curvelist.txt', 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        curvelist[0].append(float(line.split('|')[3]))
        curvelist[1].append(float(line.split('|')[5]))
        curvelist[2].append(float(line.split('|')[7]))
        
curvelist[0].append(1)
curvelist[1].append(1)
curvelist[2].append((sum(curvelist[2][-9:])+1)/10)

with open(filepath+'curvelist.pkl', 'wb') as f:
    pickle.dump(curvelist, f)
    

