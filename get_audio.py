# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 09:38:27 2019

@author: NIZI
"""
#import pyexcel as pe
import os
import subprocess as sp
import datetime as dt

def listdir(original_path, path, list_name):
    for file in os.listdir(original_path):
        file_path = os.path.join(original_path, file)  
        if os.path.isdir(file_path):  
            listdir(file_path, list_name)  
        else:  
            list_name.append(file_path)
        file_name = file.split('.')[0]
        run = 'ffmpeg -i ' + file_path + ' -f wav -vn ' + path + file_name + '.wav'
        sp.call(run, shell=True)

original_audio_path = './train_splits/'
audio_path = './train_audio/'
audio_list = []
listdir(original_audio_path, audio_path, audio_list)

