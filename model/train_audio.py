"""
Created on Jan 11 2019
Group activity label classification based on audio data.
Using transformer structure (self-attention) without any fusion models.
Experiment is based on 67 trauma cases, input samples is sentence-level data.
@author: Yue Gu, Ruiyu Zhang, Xinwei Zhao
"""

from __future__ import print_function
from keras import backend
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Masking
from keras.layers import Activation
from keras.layers import concatenate
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalMaxPooling1D
from keras.optimizers import Adam
#from data_preprocessing import data
from attention_model import AttentionLayer
from transformer import Attention, Position_Embedding
import numpy as np
from tensorflow.python import debug as tf_debug
#<meta charset="utf-8">

# Parameter setting
data_path = 'C:\\Users\\NIZI\\Documents\\a.KUKU\\UESTC\\CPSS\\1\\MELD\\MELD.Raw\\data\\'
#saving_path = r'E:/Yue/Entire Data/CNMC/'

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

#audio_input = Input(shape=(1200,92))
#audio = Masking(mask_value=0.)(audio_input)
#audio = Position_Embedding()(audio_input)
#audio = LSTM(1024,
#             return_sequences=True,
#             recurrent_dropout=0.25,
#             name='LSTM_audio_1')(audio)
#audio_vector = LSTM(256,
#             return_sequences=True,
#             recurrent_dropout=0.25,
#             name='LSTM_audio_2')(audio)
#audio_vector = GlobalMaxPooling1D()(audio_vector)
#
#d = Dense(32)(audio_vector)
#d = BatchNormalization()(d)
#d = Activation('relu')(d)
#d = Dropout(0.5)(d)
#d = Dense(16)(d)
#d = BatchNormalization()(d)
#d = Activation('relu')(d)
#
#audio_prediction = Dense(num_class, activation='softmax')(d)
#print('prediction shape: ', audio_prediction.shape)
#audio_model = Model(inputs=audio_input, outputs=audio_prediction)
#
## optimizer
#adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#audio_model.summary()


# Model Architecture
# Audio feature vector
audio_input = Input(shape=(1200, 92))
#x = Masking(mask_value=0.)(audio_input)
x = Position_Embedding()(audio_input)

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
#
#x = Attention(head_num, head_size)([x, x, x])
#x = BatchNormalization()(x)
#x = Dropout(0.15)(x)

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
d = Dropout(0.5)(d)
d = Dense(8)(d)
d = BatchNormalization()(d)
d = Activation('relu')(d)
d = Dropout(0.5)(d)
#d = Dense(16)(d)
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
#    audio_acc = 0
    # data loader (balance data)
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
#        gakki.write_epoch_acc(i, acc_a, name='Audio')
#        if acc_a >= audio_acc:
#            audio_acc = acc_a
        """
        if i >= 0:
        audio_model.save_weights(saving_path + 'audio_transformer_weights.h5')
        """
#    print('final_acc: ', audio_acc)
