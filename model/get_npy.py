# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:54:13 2019

@author: NIZI
"""

import os
import csv
import scipy.io as scio
import numpy as np
import tqdm
from keras.preprocessing import sequence

path = '../data/'

train_csvfile_path = path + 'train_sent_emo.csv'
test_csvfile_path = path + 'test_sent_emo.csv'
train_glove_path = path + 'train_sent_emo.csv'
test_glove_path = path + 'test_sent_emo.csv'
train_mfsc_path = path + 'train_MFSC/'
test_mfsc_path = path + 'test_MFSC/'
train_txt_path = path + 'train_sent_emo.txt'
test_txt_path = path + 'test_sent_emo.txt'
save_path = path
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

def readcolumn(file, txtfile):
    #table = string.maketrans("","")
    with open(file, encoding = 'utf-8') as csvfile:
        reader = csv.reader(csvfile)
        column = [row[1] for row in reader]
    column = column[1:]
    #print(column[:100])
    #print(len(column))
    f = open(txtfile, 'w+', encoding = 'utf-8')
    for line in column:
        f.write(line + '\n')
    f.close()
#    return column

def text_to_int(sentence, map_dict, max_length = 40, is_target = False):
    text_to_idx = []
    unk_idx = map_dict.get('<UNK>')
    pad_idx = map_dict.get('<PAD>')
    
    if not is_target:
        for word in sentence.lower().split():
            text_to_idx.append(map_dict.get(word, unk_idx))
    
#    else:
#        for word in sentence.lower().split():
#            text_to_idx.append(map_dict.get(word, unk_idx))
#        text_to_index.append(eos_idx)
    
    if len(text_to_idx) > max_length:
        return text_to_idx[:max_length]
    
    else:
        text_to_idx = text_to_idx + [pad_idx] * (max_length - len(text_to_idx))
        return text_to_idx

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

def gettext(savefile_name, savedict_name, txtfile_name, path):
    text = []
#    text_ = []
    readcolumn(path, txtfile_name)
    with open(txtfile_name, 'r', encoding = 'utf-8') as f:
        train_text = f.read()
    train_text = train_text[:-1]
#    train_text.strip()
    punctuations = '''!()-[]{};:'",<>.\/?@#$%^&*_~'''
    train_text.replace('...', ' ')
    for c in train_text:
        if c in punctuations:
#            print('remove')
            train_text = train_text.replace(c, "")
#    print(train_text[:100])
#    for sent in train_text:
#        res = []
#        sent.replace('...',' ') 
#        sent.replace('??','')
#        words = sent.split(' ')
#    #    print(s)
#        for word in words:
#            temp = ''
#            for c in word:
#                if c not in punctuations:
#                    temp += c
#            temp = temp.lower()
#            temp.replace(' ', '')
#            if temp != '':
#                res.append(temp)
#        text_.append(res)

#    view_sentence_range = (0, 10)
#    
#    print("-"*5 + "Training Text" + "-"*5)
#    sentences = train_text.split('\n')
#    word_counts = [len(sentence.split()) for sentence in sentences]
#    print('\nNumber of sentences: {}'.format(len(sentences)))
#    print('Average number of words in a sentence: {}'.format(np.average(word_counts)))
#    print('Max number of words in a sentence: {}'.format(np.max(word_counts)))
#    
#    print('Training sentences {} to {}'.format(*view_sentence_range))
#    print('\n'.join(sentences[view_sentence_range[0]:view_sentence_range[1]]))
    
    train_vocab = list(set(train_text.lower().split()))
#    print("The size of English vocab is : {}".format(len(train_vocab)))
    
    TRAIN_CODES = ['<PAD>', '<UNK>']
    
    train_vocab_to_int = {word: idx for idx, word in enumerate(TRAIN_CODES + train_vocab)}
#    print(train_vocab_to_int)
#    print('The size of Train dataset is : {}'.format(len(train_vocab_to_int)))
    Tx = 40
    
    for sentence in tqdm.tqdm(train_text.split('\n')):
        text.append(text_to_int(sentence, train_vocab_to_int, Tx, is_target = False))
    text = np.asarray(text)
#    print(len(text))
#    print(len(text[0]))
#    print(text)
    np.save(save_path + savedict_name, train_vocab_to_int)
    np.save(save_path + savefile_name, text)
    print('The length of ')

#def gettext(savefile_name, path):
#    text_ = []
#    text = []
#    with open(path, encoding = 'utf-8') as csvfile:
#        reader = csv.reader(csvfile)
#        column = [row[1] for row in reader]
#    column = column[1:]
##    print(column)
#    punctuations = '''!()-[]{};:'",<>.\/?@#$%^&*_~'''
#    for sent in column:
#        res = []
#        sent.replace('...',' ') 
#        sent.replace('??','')
#        words = sent.split(' ')
#    #    print(s)
#        for word in words:
#            temp = ''
#            for c in word:
#                if c not in punctuations:
#                    temp += c
#            temp = temp.lower()
#            temp.replace(' ', '')
#            if temp != '':
#                res.append(temp)
#        text_.append(res)
#    my_dict = get_dictionary()
##    length = get_max_sent_length(text_)
##    print(length)
##    t = 0
##    p = {}
##    for i in range(69):
##        p[i] = 0
#    index = 0
#    for line in text_:
#        index += 1
#        vector = []
#        #[0,0,0,...,0]
#        for i in range(200): 
#            vector.append(0)
#        #[200]
##        l = 0
#        for word in line:
#            if(word in my_dict):
#                
##                l += 1
##                my_dict[word] = [float(x) for x in my_dict[word]]
##                vector.append(my_dict[word])
#                
##                my_dict[word] = np.asarray(my_dict[word])
##                print(len(my_dict[word]))
#                
#                for i in range(len(my_dict[word])):
#                    vector[i] += float(my_dict[word][i])
#        text.append(vector)
#    text = np.asarray(text)
##        print('No.', index)
##        print('Length: ', len(vector))
##        vector = np.asarray(vector)
#        
##        if vector:
##            vector = np.pad(vector,((0, abs(40-len(vector))),(0, 0)),'constant',constant_values=(0,0))
##        else:
##            vector = np.zeros((40, 200))
##        text.append(vector[:40])
#        
##        print(len(text))
##        print(l)
##        if l > t:
##            print('l: ', l, 't: ', t)
##            t = l
##        p[l] = p[l] + 1
##    text = text[:40]
##    print(p)
##    print(len(my_dict))
##    print(len(text[0]))
##    print(len(text))
#    print(text.shape)
#    np.save(save_path + savefile_name, text)

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

#print('saving train text')
#gettext('train_text.npy', 'train_vocab_to_int.npy', train_txt_path, train_glove_path)
#print('saving test text')
#gettext('test_text.npy', 'test_vocab_to_int.npy', test_txt_path, test_glove_path)

#print('saving train audio')
#getaudio('train_audio.npy', train_mfsc_path)
#print('saving test audio')
#getaudio('test_audio.npy', test_mfsc_path)

loadData = np.load(save_path + 'train_label.npy')
print("----type----")
print(type(loadData))
print("----shape----")
print(loadData)