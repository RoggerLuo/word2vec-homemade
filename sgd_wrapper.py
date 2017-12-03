import numpy as np
import random
import skipgram
import db_model
import wv


def word2vec_sgd_wrapper(entry):
    sampleNum = 4
    cost = 0
    windowLength = 3
    step = 0.3
    trainingPairs, tokens, wordVectors = wv.getDataset(entry[2], windowLength)
    for pair in trainingPairs:
        centerword = pair[0]
        contextWords = pair[1]
        _cost = skipgram.run(centerword, contextWords, sampleNum, step)
        cost += _cost
    print(cost)


for i in range(100):
    print('第 %d 次运行' % (i,))
    version = 0
    entry = db_model.fetch_entry_untreated(version)
    for j in range(10):
        word2vec_sgd_wrapper(entry)
    db_model.mark_entry_as_treated(entry[0], version)
