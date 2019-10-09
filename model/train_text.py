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
from keras.layers import Conv1D, GlobalMaxPooling1D, Lambda
from keras.optimizers import Adam
#from data_preprocessing import data
from attention_model import AttentionLayer
from transformer import Attention, Position_Embedding
import numpy as np
from tensorflow.python import debug as tf_debug
#<meta charset="utf-8">

# Parameter setting
data_path = 'C:\\Users\\NIZI\\Documents\\a.KUKU\\UESTC\\CPSS\\1\\MELD\\MELD.Raw\\data\\'

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

num_class = 7
epoch = 10
batch_size = 8
#head_num = 8
head_size = 16
phase_1_trainable = False

# input and its shape
text_input = Input(shape = (9989,200), name = 'ph1_input')
# word embedding
em_text = Embedding(input_dim = 1470, 
                    output_dim = 200, 
                    input_length = 9989, 
                    trainable = True)(text_input)
# masking layer
text = Masking(mask_value = 0., name = 'ph1_mask')(em_text)
# LSTM layer
text = LSTM(512,
            return_sequences = True,
            recurrent_dropout = 0.25,
            name = 'ph1_LSTM_text_1')(text)
text = LSTM(256,
            return_sequences = True,
            recurrent_dropout = 0.25,
            name = 'ph1_LSTM_text_2')(text)
# batch normalization
#text_l1 = BatchNormalization(name=)(text_l1)
# attention layer
text_weight = AttentionLayer(name = 'ph1_att')(text)
text_weight = Lambda(weight_expand, name = 'ph1_lam1')(text_weight)
text_vector = Lambda(weight_dot, name = 'ph1_lam2')([text, text_weight])
text_feature_vector = Lambda(lambda x: backend.sum(x, axis = 1), name = 'ph1_lam3')(text)
# dropout layer
dropout_text = Dropout(0.25, name = 'ph1_drop1')(text_feature_vector)
dense_text_1 = Dense(128, activation = 'relu', name = 'ph1_dense')(dropout_text)
dropout_text = Dropout(0.25, name = 'ph1_drop2')(dense_text_1)
# decision-making
text_prediction = Dense(num_class, activation = 'softmax', name = 'ph1_dec')(dropout_text)
text_model = Model(inputs = text_input, outputs = text_prediction, name = 'ph1_model')
#inter_text = Model(inputs = text_input, outputs = text_feature_vector)
# optimizer
adam = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
text_model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
text_model.summary()

if __name__ == "__main__":
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
    
    for i in range(epoch):
        
        print('audio branch, epoch: ', str(i))
        text_model.fit(train_text,
                        train_label,
                        batch_size=batch_size,
                        epochs=1,
                        verbose=1)

        loss_a, acc_a = text_model.evaluate(test_text,
                                             test_label,
                                             batch_size=batch_size,
                                             verbose=0)

        print('epoch: ', str(i))
        print('loss_a', loss_a, ' ', 'acc_a', acc_a)
        print('acc_a ', acc_a)
#        gakki.write_epoch_acc(i, acc_a, name='Audio')
#        if acc_a >= audio_acc:
#            audio_acc = acc_a
        """
        if i >= 0:
        audio_model.save_weights(saving_path + 'audio_transformer_weights.h5')
        """
#    print('final_acc: ', audio_acc)
