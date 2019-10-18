# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:46:49 2019

@author: 90531
"""

import csv
import numpy as np
import tqdm
from keras.layers import Embedding

path = '../data/train_sent_emo.csv'
txtfile = '../data/train_sent_emo.txt'
file = '../data/vectors.txt'

def readcolumn(file):
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

def pretrained_embedding_layer(word_to_vec_map, source_vocab_to_int):
    vocab_len = len(source_vocab_to_int) + 1
    emb_dim = word_to_vec_map["you"].shape[0]
    
    emb_matrix = np.zeros((vocab_len,emb_dim))
    
    for word, index in source_vocab_to_int.items():
        word_vector = word_to_vec_map.get(word, np.zeros(emb_dim))
        emb_matrix[index, : ] = word_vector
    
    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)
    
    embedding_layer.build((None,))
    
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

#train_text = readcolumn(path)
#print(train_text)
readcolumn(path)
with open(txtfile, 'r', encoding = 'utf-8') as f:
    train_text = f.read()
train_text = train_text[:-1]
view_sentence_range = (0, 10)

print("-"*5 + "Training Text" + "-"*5)
sentences = train_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('\nNumber of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))
print('Max number of words in a sentence: {}'.format(np.max(word_counts)))

print('Training sentences {} to {}'.format(*view_sentence_range))
print('\n'.join(sentences[view_sentence_range[0]:view_sentence_range[1]]))

train_vocab = list(set(train_text.lower().split()))
print("The size of English vocab is : {}".format(len(train_vocab)))

TRAIN_CODES = ['<PAD>', '<UNK>']

train_vocab_to_int = {word: idx for idx, word in enumerate(TRAIN_CODES + train_vocab)}
print(train_vocab_to_int)
print('The size of Train dataset is : {}'.format(len(train_vocab_to_int)))

Tx = 40
train_text_to_int = []

for sentence in tqdm.tqdm(train_text.split('\n')):
    train_text_to_int.append(text_to_int(sentence, train_vocab_to_int, Tx, is_target = False))
train_text_to_int = np.asarray(train_text_to_int)

random_index = 0
print('-'*5 + 'Train example' + '-'*5)
#print(train_text.split('\n')[random_index])
#print(train_text_to_int[random_index])
print(len(train_text_to_int))

with open(file, 'r') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
        line = line.strip().split()
        curr_word = line[0]
        words.add(curr_word)
        word_to_vec_map[curr_word] = np.asarray(line[1:], dtype = np.float64)

embedding_layer = pretrained_embedding_layer(word_to_vec_map, train_vocab_to_int)
print(len(words))
#print(word_to_vec_map)

