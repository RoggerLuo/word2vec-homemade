import numpy as np
import random
import wv
import json
import db_model
import negSampling
from q2_sigmoid import sigmoid, sigmoid_grad


def update_o_grad(entry, grad, step):
    vec = np.array(json.loads(entry[2]))
    zeroArr = np.zeros(int(len(vec) / 2))
    vec_grad = np.concatenate((zeroArr, np.array(grad)), axis=0)
    return vec - vec_grad * step


def update_i_grad(entry, grad, step):
    vec = np.array(json.loads(entry[2]))
    zeroArr = np.zeros(int(len(vec) / 2))
    vec_grad = np.concatenate((np.array(grad), zeroArr), axis=0)
    return vec - vec_grad * step


def run(centerword, contextWords, sampleNum, step):
    centerword_entry = db_model.getWordEntrys(centerword)[0]
    centerword_vector = json.loads(centerword_entry[2])
    cost = 0.0
    gradIn = []
    # print('centerword:', centerword)

    for targetword in contextWords:
        # print('___targetword:', targetword)
        targetword_vector = json.loads(db_model.getWordEntrys(targetword)[0][2])
        negSamples_entrys = db_model.getNegSameples(contextWords, sampleNum)
        ___cost, ___cen_i_grad, ___negSamples_grad, ___target_o_grad = negSampling.get_cost_and_grad(
            centerword_vector, targetword_vector, negSamples_entrys)

        cost += ___cost
        if len(gradIn) == 0:
            gradIn = ___cen_i_grad
        else:
            gradIn += ___cen_i_grad

        for index in range(len(negSamples_entrys)):
            curr_entry = negSamples_entrys[index]
            curr_grad = ___negSamples_grad[index]
            curr_vec = update_o_grad(curr_entry, curr_grad, step)
            db_model.update_vec(curr_entry, curr_vec)

    i_vec = update_i_grad(centerword_entry, gradIn, step)
    db_model.update_vec(centerword_entry, i_vec)

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
