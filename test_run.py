#!/usr/bin/env python
#coding:utf-8
import random
import numpy as np
# from utils.treebank import StanfordSentiment
import time
from sgd_wrapper import *
from sgd import *
from datasetFactory import getdataset
from negSamplingCostAndGradient import negSamplingCostAndGradient

import db_model
import wv

# Context size
C = 3 #5

entry = db_model.fetch_entry_untreated()
string = entry[2]
entryId = entry[0]
trainingPairs, tokens, wordVectors = wv.getDataset(string,C)

# 最后mark
db_model.mark_entry_as_treated(entryId)


# Reset the random seed to make sure that everyone gets the same results
# random.seed(314)
# dataset = getdataset() # StanfordSentiment()
# tokens = dict([('我',0),('是',1),('测试',2),('句子',3),('初级',4),('前端',5),('工程师',6)]) #dataset.tokens()

# nWords = len(tokens)
# We are going to train 10-dimensional vectors for this assignment
# dimVectors = 16 #10


# Reset the random seed to make sure that everyone gets the same results
# random.seed(31415)
# np.random.seed(9265)

startTime=time.time()


# 意思是 input vector是随机启动， output vector是zeros
# randomStartVector = (np.random.rand(nWords, dimVectors) - 0.5)
# zerosVector = np.zeros((nWords, dimVectors))
# wordVectors = np.concatenate((randomStartVector/dimVectors, zerosVector),axis=0)
# print(wordVectors)

# 训练，关键
wordVectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,negSamplingCostAndGradient),
    wordVectors,
    0.3, # step 0.3
    2000, # iteration: 40000
    None,
    True,
    PRINT_EVERY=10
) 

# 这里是什么鬼，又拼接回去??? 我把它倒过来
# concatenate the input and output word vectors
# print('wordVectors before')
# print(wordVectors)
# wordVectors = np.concatenate(
#     (wordVectors[:nWords,:], wordVectors[nWords:,:]),
#     axis=0)
wordVectors = np.concatenate(
    ( wordVectors[nWords:,:],wordVectors[:nWords,:]),
    axis=0)

# wordVectors = wordVectors[:nWords,:] + wordVectors[nWords:,:]
# print('wordVectors after')
# print(wordVectors)


visualizeWords = ['我','是', '测试', '句子','前端', '工程师']#"annoying"
# print( visualizeWords[2])
visualizeIdx = [tokens[word] for word in visualizeWords]
# print('visualizeIdx')
# print(visualizeIdx)
visualizeVecs = wordVectors[visualizeIdx, :]
# print('visualizeVecs')
# print(visualizeVecs.shape)
#总之很奇怪的处理方法

# 减去平均值, 标准化吗
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
print('temp')
print(temp.shape)

# covariance 协方差
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
print('covariance')
print(covariance.shape)

# 奇异值分解
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])
print('coord')
print(coord)

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
for i in range(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
        bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('q3_word_vectors.png')
plt.show()
