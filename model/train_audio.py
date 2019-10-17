# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:19:11 2019

@author: NIZI
"""

from __future__ import print_function
from keras import backend
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Masking, Activation, concatenate, BatchNormalization
from keras.layers import Dropout, Lambda, GlobalMaxPooling1D
from keras.optimizers import Adam
from attention_model import AttentionLayer, Self_Attention
from transformer import Attention, Position_Embedding
import numpy as np
from tensorflow.python import debug as tf_debug

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
epoch = 10
batch_size = 8
head_num = 8
head_size = 16

# Audio feature vector
audio_input = Input(shape=(1200, 92))
#x = Masking(mask_value=0.)(audio_input)
x = Position_Embedding()(audio_input)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)
x = Dropout(0.15)(x)

x = Attention(head_num, head_size)([x, x, x])
x = BatchNormalization()(x)

audio_vector = GlobalMaxPooling1D()(x)
# merge layer
#print('audio_vector:', audio_vector.shape)

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
prediction = Dense(num_class, activation='softmax')(d)
print('prediction shape: ', prediction.shape)
audio_model = Model(inputs=audio_input, outputs=prediction)

# optimizer
adam = Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
audio_model.summary()

if __name__ == "__main__":
    # Audio model training
    # data loader
    train_label, train_audio, test_label, test_audio = get_data(data_path)
#    print('train_label shape: ', train_label.shape)
#    print('train_text shape: ', train_text.shape)
#    print('train_audio shape: ', train_audio.shape)
#    print('test_label shape: ', test_label.shape)
#    print('test_text shape: ', test_text.shape)
#    print('test_audio shape: ', test_audio.shape)
    
    for i in range(epoch):
        
        print('audio branch, epoch: ', str(i))
        audio_model.fit(train_audio,
                        train_label,
                        batch_size=batch_size,
                        epochs=1,
                        verbose=1)

        loss_a, acc_a = audio_model.evaluate(test_audio,
                                             test_label,
                                             batch_size=batch_size,
                                             verbose=0)

        print('epoch: ', str(i))
        print('loss_a', loss_a, ' ', 'acc_a', acc_a)
        print('acc_a ', acc_a)
