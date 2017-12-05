#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:04:43 2017

@author: codeplay2017
"""
import numpy as np
from scipy import signal
import scipy.interpolate as si
import matplotlib.pyplot as plt
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

def devide_data(original, lengthOfEachSample, divideGap, piecesDivided):
    origninal = np.array(original)
    originalNumber, length = original.shape
    numOfPieces = originalNumber*piecesDivided
    
    if (piecesDivided-1)*divideGap+lengthOfEachSample >= length:
        piecesDivided = (length-lengthOfEachSample)//divideGap
    
    y = np.zeros(lengthOfEachSample, numOfPieces)
    
    for kk in range(originalNumber*piecesDivided):
        ii = kk//piecesDivided + 1
        istart = (kk - (ii-1)*piecesDivided - 1)*divideGap + 1        
    

def get_angle(time, velocity):
    '''
    prepare for interpolate the time sequence
    input: time and velocity must have the same dimension
            'velocity' should be a sequence of angular velocity according to 'time'
    return: angle sequence
    '''
    N = time.shape[0]
    angle = np.zeros([N])
    for ii in range(1,N):
        # integrate the velocity to obtain angle
        angle[ii] = angle[ii-1] + (time[ii]-time[ii-1])*velocity[ii]
    return angle

def preprocess_angular_resample(filename,
               lengthOfEachSample,
               step,
               method='cwt',
               num_pieces=3000,
               save_image=True):
    sig, t, fs, data_type, ro_speed = load_data(filename)
    N = sig.shape[0]
    velocity = np.array([ro_speed*np.pi]*N) # angular velocity at each time point
    t2a = get_angle(t, velocity)
    min_velocity = 10*np.pi # minimum velocity in 5 conditions
#    As = 1/fs*np.min(velocity) # angular resample interval
    As = 1/fs*min_velocity # angular resample interval
    new_angle = np.linspace(0, (N-1)*As, N)
        
        

if __name__ == '__main__':
    sig, t, fs, data_type, ro_speed = load_data('/home/codeplay2017/code/lab/data/TestDataFromWen/wwg-20150407-PlanetMissTooth/20150407pmt_12k_50.txt')
    N = sig.shape[0]
    velocity = np.array([int(ro_speed)*np.pi]*N) # angular velocity at each time point
    t2a = get_angle(t, velocity)
    min_velocity = 10*np.pi # minimum velocity in 5 conditions
#    As = 1/fs*np.min(velocity) # angular resample interval
    As = 1/fs*min_velocity # angular resample interval
    new_angle = np.linspace(0, (N-1)*As, N)
    an_sig = np.zeros([N])
    t_length = 10000
#    for ii in range(20):
#        index0 = list(range(ii//2*t_length,(ii+1)//2*t_length))
#        index1 = list(range(ii*t_length/2,(ii+1)*t_length/2))
#        an_sig[index1] = si.spline(t2a[index0], sig[index0], new_angle[index0])
    an_sig = si.spline(t2a[:10000], sig[:10000], new_angle[:10000])
        
        
