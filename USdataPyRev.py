# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:48:40 2020

#%%

@author: Jiaxing
"This piece of code is the hub file that converts MATLAB data files to Python format"""

import numpy as np
from matplotlib import pylab as plt
import h5py
import os
import torch

def rgb2gray(rgb):
    return np.dot(rgb[:,:3,:,:], [0.2989, 0.5870, 0.1140])

#f = h5py.File('USDBgray.mat','r') 
f = h5py.File('data/USdata224PYrev.mat','r') 
#f = h5py.File('USdata224PY.mat','r') 

# =============================================================================

USimg500Hz = f.get('USdataAug') 
USimg500Hz = np.array(USimg500Hz)
#USimg500Hz = np.transpose(USimg500Hz, (3,2,1,0))
USimg500Hz = torch.tensor(USimg500Hz/255, dtype = torch.float32) 
USimg500HzAdd = USimg500Hz[0:2204,:,:,:]
USimg500Hz = USimg500Hz[2204::,:,:,:]


label500Hz = f.get('label500Hz')
label500Hz = np.array(label500Hz)
label500Hz = torch.tensor(label500Hz, dtype = torch.int16) 
label500HzAdd = label500Hz[0:2204]
label500Hz = label500Hz[2204::]

plateIdx = f.get('plateIdxExt') 
plateIdx = np.array(plateIdx)
plateIdx = torch.tensor(plateIdx, dtype = torch.int16) 
plateIdxAdd = plateIdx[0:2204]
plateIdx = plateIdx[2204::]

NUM_IMG = int(len(plateIdx)/6)

USimg500HzOrig = USimg500Hz[0 : NUM_IMG]
label500HzOrig = label500Hz[0 : NUM_IMG]
plateIdxOrig = plateIdx[0 : NUM_IMG]
#
# =============================================================================

#p = h5py.File('USimgOK224.mat','r') 
USimgOK = f.get('USimgGrayOK')
USimgOK = np.array(USimgOK)
#USimgOKgray = np.int16(rgb2gray(USimgOK))
USimgOKgray = torch.tensor(USimgOK/255, dtype = torch.float32) 
USimgOKgrayCh2 = torch.cat((USimgOKgray, USimgOKgray), dim = 1)
USimgOKgray = torch.cat((USimgOKgrayCh2, USimgOKgray), dim = 1)
#from imblearn.over_sampling import KMeansSMOTE
#X_resampled, y_resampled =  KMeansSMOTE().fit_resample(USimgOK, np.ones())

# =============================================================================
index = np.arange(len(label500Hz)-NUM_IMG)  ####
np.random.shuffle(index)
index = torch.tensor(NUM_IMG+index, dtype = torch.int64)
index = index[0:int(len(label500Hz)/4)]

USimg500HzAug = USimg500Hz[index]
plateIdxAug = plateIdx[index]
label500HzAug = label500Hz[index]

USimg500Hz = torch.cat((USimg500HzOrig, USimg500HzAug), dim = 0)
USimg500Hz = torch.cat((USimg500Hz, USimg500HzAdd), dim = 0)
label500Hz = torch.cat((label500HzOrig, label500HzAug), dim = 0)
label500Hz = torch.cat((label500Hz, label500HzAdd), dim = 0)
plateIdx = torch.cat((plateIdxOrig, plateIdxAug), dim = 0)
plateIdx = torch.cat((plateIdx, plateIdxAdd), dim = 0)

USimg500HzCh2 = torch.cat((USimg500HzOrig, USimg500HzOrig), dim = 1)
USimg500Hz = torch.cat((USimg500HzCh2, USimg500HzOrig), dim = 1)

USimg500HzOrigCh2 = torch.cat((USimg500HzOrig, USimg500HzOrig), dim = 1)
USimg500HzOrig = torch.cat((USimg500HzOrigCh2, USimg500HzOrig), dim = 1)

#pickle.dump([USimg500Hz, label500Hz, plateIdx], open("US_DB_Aug.p", "wb" ))
#pickle.dump([USimg500HzOrig, label500HzOrig, plateIdxOrig], open("US_DB_orig.p", "wb" ), protocol=4)