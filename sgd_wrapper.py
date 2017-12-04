import numpy as np
import random
import skipgram
import json
import db_model
import wv


def neg_entry2list(negSamples_entrys):
    returnList = []
    for entry in negSamples_entrys:  # 需要20毫秒
        returnList.append(
            {'id': entry[0], 'vector': np.array(json.loads(entry[2]))})
    return returnList


def word2vec_sgd_wrapper(entry):
    sampleNum = 10
    cost = 0
    windowLength = 3
    step = 0.3
    trainingPairs, tokens, wordVectors = wv.getDataset(entry[2], windowLength)
    for pair in trainingPairs:
        centerword = pair[0]
        contextWords = pair[1]
        negSamples_entrys = db_model.getNegSameples(
            contextWords, sampleNum)  # 平均 10 ms
        negSamples_list = neg_entry2list(negSamples_entrys)
        _cost = skipgram.run(centerword, contextWords, step, negSamples_list)
        cost += _cost
    print(cost)

version = 0
for i in range(100):
    print('第 %d 次运行' % (i,))
    entry = db_model.fetch_entry_untreated(version)
    for j in range(20):
        word2vec_sgd_wrapper(entry)
    db_model.mark_entry_as_treated(entry[0], version)
