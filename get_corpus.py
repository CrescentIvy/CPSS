# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:15:50 2019

@author: NIZI
"""

import csv
import string

csvfilename = 'train_sent_emo.csv'
txtfilename = 'Corpus.txt'
#table = string.maketrans("","")
with open(csvfilename, encoding = 'utf-8') as csvfile:
    reader = csv.reader(csvfile)
    column = [row[1] for row in reader]

column = column[1:]
#print(column[:100])
#print(len(column))

punctuations = '''!()-[]{};:'",<>.\/?@#$%^&*_~'''
res = ""

for s in column:
    s.replace('...',' ')
#    print(s)
    for c in s:
#        print(c)
        if c not in punctuations:
            res = res + c
    res += ' '
#print(res)

file = open(txtfilename,'w', encoding = 'gb18030');
file.write(str(res));
file.close();