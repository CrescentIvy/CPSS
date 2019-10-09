# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:15:50 2019

@author: NIZI
"""

import csv
import re

path = 'C:\\Users\\NIZI\\Documents\\a.KUKU\\UESTC\\CPSS\\1\\MELD\\MELD.Raw\\'
csvfilename_train = path + 'train\\train_sent_emo.csv'
csvfilename_test = path + 'test\\test_sent_emo.csv'
txtfilename = path + 'data\\Corpus.txt'

def readcolumn(file):
    #table = string.maketrans("","")
    with open(file, encoding = 'utf-8') as csvfile:
        reader = csv.reader(csvfile)
        column = [row[1] for row in reader]
    column = column[1:]
    #print(column[:100])
    #print(len(column))
    return column

train_data = readcolumn(csvfilename_train)
test_data = readcolumn(csvfilename_test)
train_data.extend(test_data)
data = train_data
#print(data)

punctuations = '''!()-[]{};:'",<>.\/?@#$%^&*_~'''
res = ""

for s in data:
    s.replace('...',' ')
#    print(s)
    for c in s:
#        print(c)
        if c not in punctuations:
            res = res + c
    res += ' '
    res = res.lower()
res = re.sub(' +', ' ', res)
#print(res)

file = open(txtfilename,'w', encoding = 'gb18030');
file.write(str(res));
file.close();