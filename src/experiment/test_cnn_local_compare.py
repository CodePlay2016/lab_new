#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 10:31:06 2017

@author: codeplay2017
"""

import numpy as np
import os, glob
import matplotlib.pyplot as plt
from PIL import Image
#import matplotlib as mp

FLAGS = None

data_num = 8
source_dir = '/home/codeplay2017/code/lab/code/paper/realwork/image/trainset9_0-50/compare_fft/'
out_dir = '/home/codeplay2017/code/lab/code/paper/realwork/python/'
model_path = os.path.join(out_dir, 'resources/models/model_2017-10-21_21:44:45/model.ckpt')

def main(): # _ means the last param
    # obersavation test
    
    f_length = 3000
    
    result = np.zeros([2400, f_length])
    a = []
    for jj in range(f_length):
        img_file_name = 'pgmt,fft,rspeed-0-50,sfre-12000_'+str(jj+1)+'.png'
        img, img_array, img_type, _, save_dir = load_img2array(
                source_dir, img_file_name)
        img_line = np.uint8(img_array)[0, 0:2400]
        result[:,jj] = img_line
    
#    Image.fromarray(result, mode='RGB').save(os.path.join(out_dir,'result_unnomed_'+img_type+'_.png'))
    plt.imshow(result, origin='lower')
    return result, a
            
#------custom functions----------------------------------
    
        
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
    result,a = main()
    
