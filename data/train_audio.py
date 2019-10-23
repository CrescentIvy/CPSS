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
#from attention_model import AttentionLayer
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
    train_audio = np.load(path + 'train_audio.npy')
    test_label = np.load(path + 'test_label.npy')
    test_audio = np.load(path + 'test_audio.npy')
    print('Finish loading data......')
    if(np.any(np.isnan(train_label)) | np.any(np.isnan(train_label)) | np.any(np.isnan(train_label)) | np.any(np.isnan(train_label)) ):
        print('Having NaN data!')
    return train_label, train_audio, test_label, test_audio

num_class = 7
epoch = 20
batch_size = 32
head_num = 8
head_size = 16

# Model Architecture
# Audio feature vector
audio_input = Input(shape=(1200, 92))
x = Masking(mask_value=0.)(audio_input)
x = Position_Embedding()(audio_input)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)
x = Dropout(0.15)(x)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)
x = Dropout(0.15)(x)
#
#x = Attention(head_num, head_size)([x, x, x])
#x = BatchNormalization()(x)
#x = Dropout(0.15)(x)
#
#x = Attention(head_num, head_size)([x, x, x])
#x = BatchNormalization()(x)
#x = Dropout(0.15)(x)
#
#x = Attention(head_num, head_size)([x, x, x])
#x = BatchNormalization()(x)
#x = Dropout(0.15)(x)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)

audio_vector = GlobalMaxPooling1D()(x)
# merge layer
print('audio_vector:', audio_vector.shape)

# decision-making
d = Dense(32)(audio_vector)
d = BatchNormalization()(d)
d = Activation('relu')(d)
d = Dropout(0.5)(d)
d = Dense(16)(d)
d = BatchNormalization()(d)
d = Activation('relu')(d)
#d = Dropout(0.5)(d)
#d = Dense(8)(d)
#d = BatchNormalization()(d)
#d = Activation('relu')(d)
#d = Dropout(0.5)(d)
#d = Dense(16)(d)
#d = BatchNormalization()(d)
#d = Activation('relu')(d)
prediction = Dense(num_class, activation='softmax')(d)
print('prediction shape: ', prediction.shape)
audio_model = Model(inputs=audio_input, outputs=prediction)

# optimizer
adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
audio_model.summary()

if __name__ == "__main__":
    history = LossHistory()
    # Audio model training
    # data loader
    train_label, train_audio, test_label, test_audio = get_data(data_path)
#    print('train_label shape: ', train_label.shape)
#    print('train_text shape: ', train_text.shape)
#    print('train_audio shape: ', train_audio.shape)
#    print('test_label shape: ', test_label.shape)
#    print('test_text shape: ', test_text.shape)
#    print('test_audio shape: ', test_audio.shape)
    
    audio_model.fit(train_audio,
                    train_label,
                    batch_size = batch_size,
                    epochs = epoch,
                    verbose = 1,
                    validation_data = (test_audio, test_label),
                    callbacks = [history])

    loss_a, acc_a = audio_model.evaluate(test_audio,
                                         test_label,
                                         batch_size = batch_size,
                                         verbose = 0)

#    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
    history.loss_plot('epoch')