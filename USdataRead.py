# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:48:40 2020



@author: Jiaxing
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

mat = loadmat(os.path.join(os.getcwd(),'data/USimgV2.mat'))

# USIpipeNG = torch.tensor(mat['USIpipeNG'])
# USIpipeNG = torch.unsqueeze(USIpipeNG,2)
# USIpipeNGch2 = torch.cat((USIpipeNG, USIpipeNG), dim = 2)
# USIpipeNG = torch.cat((USIpipeNGch2, USIpipeNG), dim = 2)

# USIpipeOK = torch.tensor(mat['USIpipeOK'])
# USIpipeOK = torch.unsqueeze(USIpipeOK,2)
# USIpipeOKch2 = torch.cat((USIpipeOK, USIpipeOK), dim = 2)
# USIpipeOK = torch.cat((USIpipeOKch2, USIpipeOK), dim = 2)

# Convert USIpipeNG to proper RGB format
USIpipeNG = torch.tensor(mat['USIpipeNG'])
# Check if it needs permutation (if it's in format H, W, N)
if USIpipeNG.dim() == 3 and USIpipeNG.shape[2] > USIpipeNG.shape[0]:
    USIpipeNG = USIpipeNG.permute(2, 0, 1)  # (N, H, W)
USIpipeNG = USIpipeNG.unsqueeze(1)      # Add channel dim: (N, 1, H, W)
USIpipeNG = USIpipeNG.repeat(1, 3, 1, 1)  # RGB: (N, 3, H, W)

# Convert USIpipeOK to proper RGB format
USIpipeOK = torch.tensor(mat['USIpipeOK'])
# Check if it needs permutation (if it's in format H, W, N)
if USIpipeOK.dim() == 3 and USIpipeOK.shape[2] > USIpipeOK.shape[0]:
    USIpipeOK = USIpipeOK.permute(2, 0, 1)  # (N, H, W)
USIpipeOK = USIpipeOK.unsqueeze(1)      # Add channel dim: (N, 1, H, W)
USIpipeOK = USIpipeOK.repeat(1, 3, 1, 1)  # RGB: (N, 3, H, W)

USIpipeNGlabel = torch.tensor(mat['USIpipelabelNG'])
USIpipeOKlabel = torch.tensor(mat['USIpipelabelOK'])
USIpipeNGidx = torch.tensor(mat['USIpipeidxNG'])
USIpipeOKidx = torch.tensor(mat['USIpipeidxOK'])


# USIplateOK = torch.tensor(mat['USIplateOK'])
# USIplateOKch2 = torch.cat((USIplateOK, USIplateOK), dim = 2)
# USIplateOK = torch.cat((USIplateOKch2, USIplateOK), dim = 2)

# USIplateNG = torch.tensor(mat['USIplateNG'])
# USIplateNGch2 = torch.cat((USIplateNG, USIplateNG), dim = 2)
# USIplateNG = torch.cat((USIplateNGch2, USIplateNG), dim = 2)

# Channels-first format: (num_images, channels, height, width)
USIplateNG = torch.tensor(mat['USIplateNG'])
USIplateNG = USIplateNG.permute(2, 0, 1)  # Shape: (8100, 224, 224)
USIplateNG = USIplateNG.unsqueeze(1)      # Add channel dim: (8100, 1, 224, 224)
USIplateNG = USIplateNG.repeat(1, 3, 1, 1)  # RGB: (8100, 3, 224, 224)

USIplateOK = torch.tensor(mat['USIplateOK'])
USIplateOK = USIplateOK.permute(2, 0, 1)
USIplateOK = USIplateOK.unsqueeze(1)
USIplateOK = USIplateOK.repeat(1, 3, 1, 1)

USIplateNGidx = torch.tensor(mat['USIplateIdxNG'])
USIplateOKidx = torch.tensor(mat['USIplateIdxOK'])
USIplateNGlabel = torch.tensor(mat['USIplateLabelNG'])
USIplateOKlabel = torch.tensor(mat['USIPlateLabelOK'])



# =============================================================================

# USimg500Hz = f.get('USdataAug') 
# USimg500Hz = np.array(USimg500Hz)
# #USimg500Hz = np.transpose(USimg500Hz, (3,2,1,0))
# USimg500Hz = torch.tensor(USimg500Hz/255, dtype = torch.float32) 
# USimg500HzAdd = USimg500Hz[0:2204,:,:,:]
# USimg500Hz = USimg500Hz[2204::,:,:,:]


# label500Hz = f.get('label500Hz')
# label500Hz = np.array(label500Hz)
# label500Hz = torch.tensor(label500Hz, dtype = torch.int16) 
# label500HzAdd = label500Hz[0:2204]
# label500Hz = label500Hz[2204::]

# plateIdx = f.get('plateIdxExt') 
# plateIdx = np.array(plateIdx)
# plateIdx = torch.tensor(plateIdx, dtype = torch.int16) 
# plateIdxAdd = plateIdx[0:2204]
# plateIdx = plateIdx[2204::]

# NUM_IMG = np.int(len(plateIdx)/6)

# USimg500HzOrig = USimg500Hz[0 : NUM_IMG]
# label500HzOrig = label500Hz[0 : NUM_IMG]
# plateIdxOrig = plateIdx[0 : NUM_IMG]

# #
# # =============================================================================

# #p = h5py.File('USimgOK224.mat','r') 
# USimgOK = f.get('USimgGrayOK')
# USimgOK = np.array(USimgOK)
# #USimgOKgray = np.int16(rgb2gray(USimgOK))
# USimgOKgray = torch.tensor(USimgOK/255, dtype = torch.float32) 

# #from imblearn.over_sampling import KMeansSMOTE
# #X_resampled, y_resampled =  KMeansSMOTE().fit_resample(USimgOK, np.ones())

# # =============================================================================
# index = np.arange(len(label500Hz)-NUM_IMG)  ####
# np.random.shuffle(index)
# index = torch.tensor(NUM_IMG+index, dtype = torch.int64)
# index = index[0:np.int(len(label500Hz)/4)]

# USimg500HzAug = USimg500Hz[index]
# plateIdxAug = plateIdx[index]
# label500HzAug = label500Hz[index]

# USimg500Hz = torch.cat((USimg500HzOrig, USimg500HzAug), dim = 0)
# USimg500Hz = torch.cat((USimg500Hz, USimg500HzAdd), dim = 0)
# label500Hz = torch.cat((label500HzOrig, label500HzAug), dim = 0)
# label500Hz = torch.cat((label500Hz, label500HzAdd), dim = 0)
# plateIdx = torch.cat((plateIdxOrig, plateIdxAug), dim = 0)
# plateIdx = torch.cat((plateIdx, plateIdxAdd), dim = 0)
# #pickle.dump([USimg500Hz, label500Hz, plateIdx], open("US_DB_Aug.p", "wb" ))
# #pickle.dump([USimg500HzOrig, label500HzOrig, plateIdxOrig], open("US_DB_orig.p", "wb" ), protocol=4)