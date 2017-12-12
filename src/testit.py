import glob
import numpy as np
import make_data_pai as md
import matplotlib.pyplot as plt

source_dir = '/home/codeplay2017/code/lab/code/paper/realwork/image/wen_data/raw_divided/time_series_step1_4096_5speeds/'
source_dir_ = '/home/codeplay2017/code/lab/code/paper/realwork/image/wen_data/raw_divided/angle_series_step1_4096_5speeds/'
flist = glob.glob(source_dir+'*-50,*.mat')
flist_ = glob.glob(source_dir+'*-50,*.mat')
matdata = md.prepare_data(flist[0], fft=False, mirror=False)[0]
matdata_ = md.prepare_data(flist_[0], fft=False, mirror=False)[0]
#testsize = 50000
#matdata = matdata[40000:testsize,:]
plt.figure()
plt.plot(matdata[47900,:])

