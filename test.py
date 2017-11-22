# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:32:37 2017

@author: Thomas Codeplay
"""
import numpy as np
import matplotlib.pyplot as plt 
  
from scipy import fftpack

pathname = r'D:\code\lab\data\TestDataFromWen\wwg-20150407-PlanetMissTooth\\'
filename = '20150407pmt_12k_10.txt'
f = open(pathname + filename)

channel = 3 # 选择第channel通道的数据 
data = []
for each_line in f:
    each_line = each_line[:len(each_line)-1]
    words = []
    words = each_line.split(sep='\t')
    nums = [float(x) for x in words]
    data.extend(nums)

data = np.array(data).reshape(-1,8) # 所有数据导入完毕

signal = data[:,[channel]] # 选择信号通道

fs = int(filename.split(sep='_')[1][:2]) * 1000 # 采样频率
N = len(signal) # 采样点数

hsignal = np.sqrt((fftpack.hilbert(signal))**2 + signal**2)

transformed = np.fft.fft(hsignal)
transformed = np.abs(transformed) * 2 / N
transformed = transformed[:N//2]
f = fs*np.arange(0,N//2)/N

plt.figure(1)
plt.plot(f,transformed)
plt.title(filename + '  fft') 
plt.show()

plt.figure(2)
plt.plot(signal,'r')
plt.plot(hsignal,'b',linewidth=2)

#print np.all(np.abs(np.fft.ifft(transformed)))
