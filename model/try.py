# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:59:00 2019

@author: NIZI
"""

from keras.models import Model
from keras.layers import Input,Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import GlobalMaxPooling1D
import numpy as np

data_path = 'C:\\Users\\NIZI\\Documents\\a.KUKU\\UESTC\\CPSS\\1\\MELD\\MELD.Raw\\data\\'

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
batch_size = 32
head_num = 8
head_size = 16

inputs = Input(shape=(1200,92))
hidden = Dense(units=10,activation='relu')(inputs)
output = Dense(units=5,activation='sigmoid')(hidden)
output = GlobalMaxPooling1D()(output)
d = Dense(32)(output)
d = BatchNormalization()(d)
d = Activation('tanh')(d)
d = Dropout(0.5)(d)
d = Dense(16)(d)
d = BatchNormalization()(d)
d = Activation('tanh')(d)
prediction = Dense(num_class, activation='softmax')(d)
model = Model(inputs=inputs, outputs=prediction)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

if __name__ == "__main__":
    train_label, train_audio, test_label, test_audio = get_data(data_path)
    print(train_label)
    print(train_audio)
#    for i in range(epoch):
#        print('audio branch, epoch: ', str(i))
#        model.fit(train_audio,
#                        train_label,
#                        batch_size=batch_size,
#                        epochs=1,
#                        verbose=1)
#    
#        loss_a, acc_a = model.evaluate(test_audio,
#                                       test_label,
#                                       batch_size=batch_size,
#                                       verbose=0)
#
#    print('epoch: ', str(i))
#    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
#    print('acc_a ', acc_a)

