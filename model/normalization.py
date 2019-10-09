#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:48:41 2019

@author: nizi
"""
import scipy.io as scio
import os
import numpy as np

data_path = 'C:\\Users\\NIZI\\Documents\\a.KUKU\\UESTC\\CPSS\\1\\MELD\\MELD.Raw\\'

path1 = data_path + 'train\\train_MFSC_original\\'
path2 = data_path + 'test\\test_MFSC_original\\'
#file = './result1012/L/170122/170122 (163727) transcript-0001.mat'
resultpath1 = data_path + 'train\\train_MFSC\\'
resultpath2 = data_path + 'test\\test_MFSC\\'

INTMAX = 65535
INTMIN = -65535

def findmat(path):
    files = []
    pathlist = os.listdir(path)
    pathlist.sort()
    for file in pathlist:
        files.append(file)
    return files

def findboundary(path):
    Max = INTMIN
    Min = INTMAX
    files = findmat(path)
    for file in files:
        data = scio.loadmat(path+file)
        a = np.array(data['z1'])
        where_are_inf = np.isinf(a)
        a[where_are_inf] = 0
        if(a.max() > Max):
            Max = a.max()
        if(a.min() < Min):
            Min = a.min()
#    if Min < INTMIN:
#        Min = INTMIN
#    if Max > INTMAX:
#        Max = INTMAX
    return Max,Min

def normalization_to_new_path(path,respath,Max,Min):
    files = findmat(path)
    for file in files:
        data = scio.loadmat(path + file)
        a = np.array(data['z1'])
        b = (a-Min)/(Max-Min)
        scio.savemat(respath+file,{'z1': b})

#Lmax, Lmin = findboundary(path1)
#Rmax, Rmin = findboundary(path2)
#
#if(Lmax > Rmax):
#    maxvalue = Lmax
#else:
#    maxvalue = Rmax
#if(Lmin < Rmin):
#    minvalue = Lmin
#else:
#    minvalue = Rmin
#
#print(maxvalue, minvalue)
#
#normalization_to_new_path(path1,resultpath1,maxvalue,minvalue)
#normalization_to_new_path(path2,resultpath2,maxvalue,minvalue)

files = findmat(path1)
for file in files:
    data = scio.loadmat(path1+file)
    a = np.array(data['z1'])
    if(a.dtype )
