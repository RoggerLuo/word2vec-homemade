#!/usr/bin/env python
import numpy as np
import random
import getTrainingPair
def getdataset():    
    dataset = type('dummy', (), {})() 
    def dummySampleTokenIdx():
        return random.randint(0, 4)    # 0到4中间随机选一个

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getTrainingPair.fetchOne #getRandomContext

    random.seed(31415)
    np.random.seed(9265)

    return dataset


    """
    返回两个
        1.一个字符，a到e中的一个
        2.返回一个list，长度是输入数值的2倍
    """
    # def getRandomContext(C,i):
    #     # tokens = ["a", "b", "c", "d", "e"]
    #     # return tokens[random.randint(0,4)], \
    #     #     [tokens[random.randint(0,4)] for i in range(2*C)]
    #     # wordSegmentation()
    #     if i%4 == 1:
    #         return '测试', ['前端','我','工程师','是','测试']

    #     if i%4 == 2:
    #         return '句子', ['是', '测试', '句子']

    #     if i%4 == 0:
    #         return '工程师', ['测试','我','是', '前端', '工程师']
    #     if i%4 == 3:
    #         return '前端', ['工程师','是','前端', '测试']

    # 第一个方法有什么用
    # note:这个是正态分布
    # 还是不知道这个normalizeRows有什么意义，用在正态分布上并无太大意义
    # dummy_vectors = normalizeRows(np.random.randn(10,3))

    # k v 对象
    # dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

