# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:22:32 2019

@author: NIZI
"""

from __future__ import print_function
from keras import backend
from keras.models import Model
from keras.layers import Dense, Dropout, Input, LSTM
from keras.layers import Bidirectional, Masking, Embedding, concatenate
from keras.layers import BatchNormalization, Activation, TimeDistributed
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Lambda
from keras.optimizers import Adam
from attention_model import AttentionLayer, Self_Attention
from transformer import Attention, Position_Embedding
import numpy as np
from Losshistory import LossHistory
from tensorflow.python import debug as tf_debug
#<meta charset="utf-8">

# Parameter setting
data_path = '../data/'

def weight_expand(x):
    return backend.expand_dims(x)

def weight_dot(inputs):
    x = inputs[0]
    y = inputs[1]
    return x * y

def get_data(path):
    print('Loading data.......')
    train_label = np.load(path + 'train_label.npy')
    train_text = np.load(path + 'train_text.npy')
    test_label = np.load(path + 'test_label.npy')
    test_text = np.load(path + 'test_text.npy')
    print('Finish loading data......')
    if(np.any(np.isnan(train_label)) | np.any(np.isnan(train_text)) | np.any(np.isnan(test_label)) | np.any(np.isnan(test_text)) ):
        print('Having NaN data!')
    return train_label, train_text, test_label, test_text

def get_embeded_matrix():
    glove_file = data_path + 'vectors.txt'
    dictionary = []
    with open(glove_file, 'r') as f:
        content = f.readlines()
        for line in content:
            line = line.split(' ')
            val = []
            for word in line[1:]:
                val.append(float(word.strip()))
            dictionary.append(val)
    return dictionary

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

with open(data_path + 'vectors.txt', 'r') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
        line = line.strip().split()
        curr_word = line[0]
        words.add(curr_word)
        word_to_vec_map[curr_word] = np.asarray(line[1:], dtype = np.float64)

_train_vocab_to_int = np.load(data_path + 'train_vocab_to_int.npy', allow_pickle = True)
train_vocab_to_int = _train_vocab_to_int.item()
vocab_len = len(train_vocab_to_int) + 1
emb_dim = word_to_vec_map['you'].shape[0]
#print(len(train_vocab_to_int))
#embedding_layer = pretrained_embedding_layer(word_to_vec_map, train_vocab_to_int)

num_class = 7
epoch = 20
batch_size = 16
head_num = 8
head_size = 16
phase_1_trainable = False

# input and its shape
text_input = Input(shape = (40,), name = 'ph1_input')
# word embedding
text = Embedding(vocab_len, emb_dim, trainable = False)(text_input)
#em_text = embedding_layer(text_input)
#em_text = Embedding(input_dim = 1470, 
#                    output_dim = 200, 
#                    input_length = 9989, 
#                    trainable = True)(text_input)
# masking layer
text = Masking(mask_value = 0., name = 'ph1_mask')(text)
# LSTM layer
#text = LSTM(512,
#            return_sequences = True,
#            recurrent_dropout = 0.25,
#            name = 'ph1_LSTM_text_1')(text)
#text = LSTM(200,
#            return_sequences = True,
#            recurrent_dropout = 0.25, 
#            name = 'ph1_LSTM_text_2')(text)
#text = LSTM(128,
#            recurrent_dropout = 0.25,
#            name = 'ph1_LSTM_text_3')(text)
# attention layer
#text_weight = Attention(head_num, head_size)([text, text, text])
#text_weight = AttentionLayer(name = 'ph1_att')(text)
text_weight = Self_Attention(emb_dim, name = 'ph1_att')(text)
# batch normalization
text_weight = BatchNormalization(name = 'batch_1')(text_weight)
# feed Forward
text_weight = Dropout(0.5)(text_weight)
# batch normalization
text_weight = BatchNormalization(name = 'batch_2')(text_weight)

text_weight = Self_Attention(emb_dim, name = 'ph2_att')(text_weight)
text_weight = BatchNormalization(name = 'batch_3')(text_weight)
text_weight = Dropout(0.5)(text_weight)
text_weight = BatchNormalization(name = 'batch_4')(text_weight)

#text_weight = Lambda(weight_expand, name = 'ph1_lam1')(text_weight)
text_vector = Lambda(weight_dot, name = 'ph1_lam2')([text, text_weight])
text_feature_vector = Lambda(lambda x: backend.sum(x, axis = 1), name = 'ph1_lam3')(text_vector)
# dropout layer
dropout_text = Dropout(0.25, name = 'ph1_drop1')(text_feature_vector)
dense_text_1 = Dense(64, activation = 'relu', name = 'ph1_dense')(dropout_text)
dropout_text = Dropout(0.25, name = 'ph1_drop2')(dense_text_1)
# decision-making
text_prediction = Dense(num_class, activation = 'softmax', name = 'ph1_dec')(dropout_text)
text_model = Model(inputs = text_input, outputs = text_prediction, name = 'ph1_model')
# optimizer
adam = Adam(lr = 0.0001, beta_1 = 0.99, beta_2 = 0.999, epsilon = 1e-08)
text_model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
text_model.summary()

if __name__ == "__main__":
    history = LossHistory()
    # Audio model training
#    audio_acc = 0
    # data loader (balance data)
    train_label, train_text, test_label, test_text = get_data(data_path)
    print('train_label shape: ', train_label.shape)
    print('train_text shape: ', train_text.shape)
#    print('train_audio shape: ', train_audio.shape)
#    print('test_label shape: ', test_label.shape)
#    print('test_text shape: ', test_text.shape)
#    print('test_audio shape: ', test_audio.shape)

#    for i in range(epoch):
#        
#        print('audio training branch, epoch: ', str(i))
#        text_model.fit(train_text,
#                        train_label,
#                        batch_size=batch_size,
#                        epochs=1,
#                        verbose=1,
#                        validation_data = (test_text, test_label),
#                        callbacks = [history])
#        loss_a, acc_a = text_model.evaluate(test_text,
#                                             test_label,
#                                             batch_size=batch_size,
#                                             verbose=1)
    #    print('testing epoch: ', str(i))
    #    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
    text_model.fit(train_text,
                    train_label,
                    batch_size=batch_size,
                    epochs=epoch,
                    verbose=1,
                    validation_data = (test_text, test_label),
                    callbacks = [history])
    loss_a, acc_a = text_model.evaluate(test_text,
                                         test_label,
                                         batch_size=batch_size,
                                         verbose=1)

    history.loss_plot('epoch')
    
