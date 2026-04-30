# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:48:40 2020
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

"This piece of code is the hub file that converts MATLAB data files to Python format"""

import numpy as np
from matplotlib import pylab as plt
import h5py
import os
import torch
from scipy.io import loadmat

def rgb2gray(rgb):
    return np.dot(rgb[:,:3,:,:], [0.2989, 0.5870, 0.1140])

#f = h5py.File('USDBgray.mat','r') 
# f = h5py.File(os.getcwd()+'\\USdata224PYrev.mat','r') 
#f = h5py.File('USdata224PY.mat','r') 

# f = h5py.File(os.path.join(os.getcwd(), 'data\\USimg2021.mat'), 'r')

mat = loadmat(os.path.join(os.getcwd(),'data\\USimgV2.mat'))

# =============================================================================
# preparing the data tensor
USIpipeNG = torch.tensor(mat['USIpipeNG'])/255
USIpipeNG = torch.unsqueeze(USIpipeNG,2)
USIpipeNGch2 = torch.cat((USIpipeNG, USIpipeNG), dim = 2)
USIpipeNG = torch.cat((USIpipeNGch2, USIpipeNG), dim = 2)

# Transpose the tensor to the (N, C, H, W) format
USIpipeNG = USIpipeNG.permute(3, 2, 0, 1)

USIpipeOK = torch.tensor(mat['USIpipeOK'])/255
USIpipeOK = torch.unsqueeze(USIpipeOK,2)
USIpipeOKch2 = torch.cat((USIpipeOK, USIpipeOK), dim = 2)
USIpipeOK = torch.cat((USIpipeOKch2, USIpipeOK), dim = 2)

# Transpose the tensor to the (N, C, H, W) format
USIpipeOK = USIpipeOK.permute(3, 2, 0, 1)


USIpipeNGlabel = torch.tensor(mat['USIpipelabelNG'])
USIpipeOKlabel = torch.tensor(mat['USIpipelabelOK'])-1
USIpipeNGidx = torch.tensor(mat['USIpipeidxNG'])
USIpipeOKidx = torch.tensor(mat['USIpipeidxOK'])


USIplateOK = torch.tensor(mat['USIplateOK'])/255
USIplateOK = torch.unsqueeze(USIplateOK,2)
USIplateOKch2 = torch.cat((USIplateOK, USIplateOK), dim = 2)
USIplateOK = torch.cat((USIplateOKch2, USIplateOK), dim = 2)
USIplateOK = USIplateOK.permute(3, 2, 0, 1)

USIplateNG = torch.tensor(mat['USIplateNG'])/255
USIplateNG = torch.unsqueeze(USIplateNG,2)
USIplateNGch2 = torch.cat((USIplateNG, USIplateNG), dim = 2)
USIplateNG = torch.cat((USIplateNGch2, USIplateNG), dim = 2)
USIplateNG = USIplateNG.permute(3, 2, 0, 1)

USIplateNGidx = torch.tensor(mat['USIplateIdxNG'])
USIplateOKidx = torch.tensor(mat['USIplateIdxOK'])
USIplateNGlabel = torch.tensor(mat['USIplateLabelNG'])
USIplateOKlabel = torch.tensor(mat['USIplateLabelOK'])
