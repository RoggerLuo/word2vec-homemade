import numpy as np
import random
import wv
import json
import db_model
import negSampling
from q2_sigmoid import sigmoid, sigmoid_grad


def update_o_grad(entry, grad, step):
    # global step
    vec = entry['vector']
    zeroArr = np.zeros(int(len(vec) / 2))
    vec_grad = np.concatenate((zeroArr, np.array(grad)), axis=0)
    return vec - vec_grad * step


def update_i_grad(entry, grad, step):
    vec = entry['vector']
    zeroArr = np.zeros(int(len(vec) / 2))
    vec_grad = np.concatenate((np.array(grad), zeroArr), axis=0)
    return vec - vec_grad * step


def getEntry_and_makeList(centerword):
    entry = db_model.getWordEntrys(centerword)[0]
    vec = np.array(json.loads(entry[2]))
    return {'id': entry[0], 'vector': vec}, vec


def run(centerword, contextWords, step, negSamples_list):
    assert type(centerword) == str
    assert type(contextWords) == list
    assert type(step) == float
    assert type(negSamples_list) == list

    cen_entry, cent_vec = getEntry_and_makeList(centerword)
    cost = 0.0
    gradIn = []
    for targetword in contextWords:
        target_vec = json.loads(db_model.getWordEntrys(targetword)[0][2])
        ___cost, ___cen_i_grad, ___negSamples_grad, ___target_o_grad = negSampling.get_cost_and_grad(
            cent_vec, target_vec, negSamples_list)
        cost += ___cost
        if len(gradIn) == 0:
            gradIn = ___cen_i_grad
        else:
            gradIn += ___cen_i_grad

        for index in range(len(negSamples_list)):
            curr_entry = negSamples_list[index]
            assert type(curr_entry) == dict

            curr_grad = ___negSamples_grad[index]
            assert type(curr_grad) == np.ndarray

            curr_vec = update_o_grad(curr_entry, curr_grad, step)
            assert type(curr_vec) == np.ndarray
            assert len(curr_vec) == 16

            negSamples_list[index]['vector'] = curr_vec  # 更新完才开始下一轮

    if len(gradIn) == 0:
        return 0.0

    i_vec = update_i_grad(cen_entry, gradIn, step)
    assert type(i_vec) == np.ndarray
    assert len(i_vec) == 16

    db_model.update_vec(cen_entry, i_vec)

    for sampleEntry in negSamples_list: 
        assert type(sampleEntry['vector']) == np.ndarray
        assert len(sampleEntry['vector']) == 16

        db_model.update_vec(sampleEntry, sampleEntry['vector'])

    return cost


# db_model.mark_entry_as_treated(entry[0])

# entry = db_model.fetch_entry_untreated()
# string = entry[2]
# # print(string)
# windowLength = 3
# trainingPairs, tokens, wordVectors = wv.getDataset(string, windowLength)
# pair = trainingPairs[0]
# centerword = pair[0]
# contextWords = pair[1]
# sampleNum = 4
# centerword_vector = json.loads(db_model.getWordEntrys(centerword)[0][2])


# a = skipgram(centerword, contextWords)
# print(a)
