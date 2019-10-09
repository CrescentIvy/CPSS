# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:54:13 2019

@author: NIZI
"""

import os
import csv
import scipy.io as scio
import numpy as np
from decimal import Decimal
from keras.preprocessing import sequence

path = 'C:\\Users\\NIZI\\Documents\\a.KUKU\\UESTC\\CPSS\\1\\MELD\\MELD.Raw\\'

train_csvfile_path = path + 'train\\train_sent_emo.csv'
test_csvfile_path = path + 'test\\test_sent_emo.csv'
train_glove_path = path + 'train\\train_sent_emo.csv'
test_glove_path = path + 'test\\test_sent_emo.csv'
train_mfsc_path = path + 'train\\train_MFSC\\'
test_mfsc_path = path + 'test\\test_MFSC\\'
save_path = path + 'data\\'
#table = string.maketrans("","")

#def get_embeded_matrix():
#    glove_file = path + 'data\\vectors.txt'
#    dictionary = []
#    with open(glove_file, 'r') as f:
#        content = f.readlines()
#        for line in content:
#            line = line.split(' ')
#            val = []
#            for word in line[1:]:
#                val.append(float(word.strip()))
#            dictionary.append(val)
#    return dictionary

def get_max_sent_length(text):
    length = 0
    for line in text:
        if(len(line) > length):
            length = len(line)
    return length

def get_dictionary():
    glove_file = path + 'data\\vectors.txt'
    dictionary = {}
    with open(glove_file, 'r') as f:
        content = f.readlines()
        for line in content:
            line = line.split(' ')
            key = line[0]
            val = []
            for word in line[1:]:
                val.append(word.strip())
            dictionary[key] = val
    return dictionary

def getlabel(savefile_name, csvfile_path):
    with open(csvfile_path, encoding = 'utf-8') as csvfile:
        reader = csv.reader(csvfile)
        column = [row[3] for row in reader]
    
    column = column[1:]
    #print(len(column))
    label_ = []
    for emo in column:
        l = []
        if emo == 'anger':
            l = [1, 0, 0, 0, 0, 0, 0]
            label_.append(np.asarray(l))
        elif emo == 'disgust':
            l = [0, 1, 0, 0, 0, 0, 0]
            label_.append(np.asarray(l))
        elif emo == 'fear':
            l = [0, 0, 1, 0, 0, 0, 0]
            label_.append(np.asarray(l))
        elif emo == 'joy':
            l = [0, 0, 0, 1, 0, 0, 0]
            label_.append(np.asarray(l))
        elif emo == 'neutral':
            l = [0, 0, 0, 0, 1, 0, 0]
            label_.append(np.asarray(l))
        elif emo == 'sadness':
            l = [0, 0, 0, 0, 0, 1, 0]
            label_.append(np.asarray(l))
        elif emo == 'surprise':
            l = [0, 0, 0, 0, 0, 0, 1]
            label_.append(np.asarray(l))
    label = np.asarray(label_)
    print(label)
    np.save(save_path + savefile_name, label)

def gettext(savefile_name, path):
    text_ = []
    text = []
#    english='H:\\自然语言处理\\Experiment2\\English.txt'
#    with open(english,'r',encoding='utf-8') as file:
#        u=file.read()
#    str=re.sub('[^\w ]','',u)
#    print(nltk.word_tokenize(str))
#    print(nltk.pos_tag(nltk.word_tokenize(str))) #对分完词的结果进行词性标注
    with open(path, encoding = 'utf-8') as csvfile:
        reader = csv.reader(csvfile)
        column = [row[1] for row in reader]
    column = column[1:]
#    print(column)
    punctuations = '''!()-[]{};:'",<>.\/?@#$%^&*_~'''
    for sent in column:
        res = []
        sent.replace('...',' ') 
        sent.replace('??','')
        words = sent.split(' ')
    #    print(s)
        for word in words:
            temp = ''
            for c in word:
                if c not in punctuations:
                    temp += c
            temp = temp.lower()
            temp.replace(' ', '')
            if temp != '':
                res.append(temp)
        text_.append(res)
    my_dict = get_dictionary()
#    length = get_max_sent_length(text_)
#    print(length)
#    t = 0
#    p = {}
#    for i in range(69):
#        p[i] = 0
    index = 0
    for line in text_:
        index += 1
        vector = []
        #[0,0,0,...,0]
        for i in range(200): 
            vector.append(0)
        #[200]
#        l = 0
        for word in line:
            if(word in my_dict):
                
#                l += 1
#                my_dict[word] = [float(x) for x in my_dict[word]]
#                vector.append(my_dict[word])
                
#                my_dict[word] = np.asarray(my_dict[word])
#                print(len(my_dict[word]))
                
                for i in range(len(my_dict[word])):
                    vector[i] += float(my_dict[word][i])
        text.append(vector)
    text = np.asarray(text)
#        print('No.', index)
#        print('Length: ', len(vector))
#        vector = np.asarray(vector)
        
#        if vector:
#            vector = np.pad(vector,((0, abs(40-len(vector))),(0, 0)),'constant',constant_values=(0,0))
#        else:
#            vector = np.zeros((40, 200))
#        text.append(vector[:40])
        
#        print(len(text))
#        print(l)
#        if l > t:
#            print('l: ', l, 't: ', t)
#            t = l
#        p[l] = p[l] + 1
#    text = text[:40]
#    print(p)
#    print(len(my_dict))
#    print(len(text[0]))
#    print(len(text))
    print(text.shape)
    np.save(save_path + savefile_name, text)

def getaudio(savefile_name, path):
    audio = []
#    length_ = []
    num = 0
    for f in os.listdir(path):
        num = num + 1
        file = path + f
#        print(file)
        data = scio.loadmat(file)
        data = data['z1']
        data = sequence.pad_sequences(data, padding='post', truncating='post', dtype='float32', maxlen=1200)
        # print(tmp.shape)
        data = data.transpose()
        audio_ = np.asarray(data)
#        length = len(data['z1'][0])
#        audio_ = np.asarray(data['z1'][:,:1200])
#        if length < 1200:
#            audio_ = np.pad(audio_,((0,0),(0,1200-length)),'constant')
        audio.append(audio_)
#    print(len(audio))
#    print(num)
    np.save(save_path + savefile_name, audio)

#print('saving train label')
#getlabel('train_label.npy', train_csvfile_path)
#print('saving test label')
#getlabel('test_label.npy', test_csvfile_path)

print('saving train text')
gettext('train_text.npy', train_glove_path)
print('saving test text')
gettext('test_text.npy', test_glove_path)

#print('saving train audio')
#getaudio('train_audio.npy', train_mfsc_path)
#print('saving test audio')
#getaudio('test_audio.npy', test_mfsc_path)

#loadData = np.load(save_path + 'train_audio.npy')
#print("----type----")
#print(type(loadData))
#print("----shape----")
#print(loadData.shape)