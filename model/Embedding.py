# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:19:09 2019

@author: 90531
"""

import numpy as np
from keras.layers import Embedding

file = '../data/vectors.txt'

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

with open(file, 'r') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
        line = line.strip().split()
        curr_word = line[0]
        words.add(curr_word)
        word_to_vec_map[curr_word] = np.asarray(line[1:], dtype = np.float64)

#embedding_layer = pretrained_embedding_layer(word_to_vec_map, train_vocab_to_int)