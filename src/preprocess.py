#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:04:43 2017

@author: codeplay2017
"""
import numpy as np
from scipy import signal
import glob

def load_data(filepath, channel=4):
    # filepath: like '.../.../20150407na_12k_50.txt'
    # channel: totally 8 channels, default is channel 4
    sig = np.loadtxt(filepath)[:,channel]
    filename = filepath.split('/')[-1]
    fs = int(filename.split('_')[1].split('k')[0])*1000 # sample frequency
    N = sig.shape[0] # sampled points
    ts = 1/fs # sample period
    t = np.linspace(0, (N-1)*ts, N) # corresponding timeline
    switcher = {
                "na": 'normal',
                "pmt": 'pgmt',
                "psf": 'pgsw'
                }
    data_type = switcher.get(filename.split('_')[0][8:]) # health condition
    ro_speed = filename.split('_')[-1].split('.')[0] # rotational frequency
    return sig, t, fs, data_type, ro_speed

def preprocess_angular_resample(file_list,
               lengthOfEachSample,
               step,
               method='cwt',
               num_pieces=3000,
               save_image=True):
    for file in file_list:
        sig, t, fs, data_type, ro_speed = load_data(file)
        
        
def main():
    sig,_,_,_,_ = load_data('/home/codeplay2017/code/lab/data/TestDataFromWen/wwg-20150407-PlanetMissTooth/20150407pmt_12k_50.txt')
#    signal.morlet
    return sig

if __name__ == '__main__':
    sig = main()
        
        