#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:33:00 2017

@author: codeplay2017
"""

import pickle
import matplotlib.pyplot as plt

def plot_for_model(curvelist,
                   fig_title='Accuracy curves',
                   label = 'test',
                   annotate=True,
                   color='#1f77b4ff',
                   annotate_xytext=(-30, -10)):
    iter_length = len(curvelist)
    iterlist = list(range(100,iter_length*100+1,100))
    plt.title(fig_title)
    plt.plot(iterlist, curvelist,label=label, color=color)
#    plt.plot(iterlist, curvelist[1],label=labels[1], color='#ff7f0eff')
#    plt.plot(iterlist, curvelist[2],label=labels[2], color='#2ca02cff')
    
#    plt.scatter([iterlist[train_max_index],], curvelist[2][train_max_index], 50)
#     plot the '---' above the max of last 10 test accuracy, add the label
    train_max_index = curvelist.index(max(curvelist))
    plt.plot([0, iterlist[train_max_index]], [max(curvelist), max(curvelist)],
             color='red', linestyle='--')
    if annotate:
        plt.annotate('%.3g' % (max(curvelist)), xy=(0, max(curvelist)),
                 xytext=annotate_xytext, textcoords='offset points')
    plt.plot([0, iterlist[train_max_index]], [max(curvelist), max(curvelist)],
             color='red', linestyle='--')
    
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.xlim(0,iter_length*100)
    plt.xlim(0,20000)
    plt.legend()
    ax = plt.gca()
    return ax
    
def plot_between_models(curvelists, curvename='train'):
    switcher = {
                "train": 0,
                "test": 1,
                "last 10 test": 2
                }
    curve_index = switcher.get(curvename)
    iterlist = list(range(100,20001,100))
    plt.figure()
    plt.title('Last 10 test accuracy curves from different data')
    list_num = len(curvelists)
    for ii in range(list_num):
        curvelist = curvelists[ii]
        max_index = curvelist[curve_index].index(max(curvelist[curve_index]))
        plt.plot([0, iterlist[max_index]], [max(curvelist[curve_index]), 
                 max(curvelist[curve_index])], color='red', linestyle='--')
        if ii == 0:
            plt.plot(iterlist, curvelist[curve_index], label='cwt', color='#1f77b4ff')
            plt.annotate('%.3g' % (max(curvelist[curve_index])), xy=(0, max(curvelist[curve_index])),
                     xytext=(-30, -10), textcoords='offset points')
        elif ii == 1:    
            plt.plot(iterlist, curvelist[curve_index], label='raw', color='#ff7f0eff')
            plt.annotate('%.3g' % (max(curvelist[curve_index])), xy=(0, max(curvelist[curve_index])),
                     xytext=(-30, +5), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        elif ii == 2:    
            plt.plot(iterlist, curvelist[curve_index], label='fft', color='#23ffa0ff')
            
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.xlim(0,15000)
    plt.legend()
    ax = plt.gca()
    return ax
    
def main():
    filepath1 = '/home/codeplay2017/code/lab/code/paper/realwork/python/observation/171122/raw_2048_50Hz_step1/exp1/curvelist.pkl'
    filepath2 = '/home/codeplay2017/code/lab/code/paper/realwork/python/observation/171122/fft_1024_5speeds_step1/exp1/curvelist.pkl'
    with open(filepath1, 'rb') as f:
        curvelist1 = pickle.load(f, encoding='latin1') # the 'encoding' parameter solves the compatibility between python2 and 3
    with open(filepath2, 'rb') as f:
        curvelist2 = pickle.load(f, encoding='latin1') # the 'encoding' parameter solves the compatibility between python2 and 3
        print(len(curvelist2[0]))
    
    plt.figure(1)    
    plot_for_model(curvelist1[1],annotate_xytext=(-30,-5),annotate=False)
    plot_for_model(curvelist1[2],label='last10 test',color='orange',annotate_xytext=(-30,0),annotate=False)
    plt.figure(2)
    plot_for_model(curvelist2[1],annotate=False)
    plot_for_model(curvelist2[2],label='last10 test',color='orange',annotate=False)
    
#    plot_for_model(curvelist1[2], figure=False, color='red',
#                   labels=['3-3-3,step20'],
#                   annotate=False)
#    plot_for_model(curvelist2[2], figure=False, color='pink',
#                   labels=['new3-3-3,step2400'],
#                   fig_title='Accuracy curve for different recording method',
#                   annotate=False)
#    plot_between_models(curvelists=(curvelist1, curvelist2, curvelist3), curvename='last 10 test')
    
    
if __name__ == '__main__':
    main()
    
