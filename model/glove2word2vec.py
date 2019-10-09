# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:46:04 2019

@author: NIZI
"""

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


glove_input_file = 'glove.300d.txt'
word2vec_output_file = 'glove.300d.word2vec.txt'

glove2word2vec(glove_input_file, word2vec_output_file)

# 加载模型
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, encoding = 'gbk')
# 获得单词I的词向量
#I_vec = glove_model['I']
#print(I_vec)
# 获得单词You的最相似向量的词汇
print(glove_model.most_similar('You'))