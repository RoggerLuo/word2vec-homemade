#!/usr/bin/env python
import numpy as np
import random
import wv
import json
import db_model
import negSampling
from q2_sigmoid import sigmoid, sigmoid_grad


# def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
#              dataset, word2vecCostAndGradient):

#     cost = 0.0
#     gradIn = np.zeros(inputVectors.shape)
#     gradOut = np.zeros(outputVectors.shape)

#     # YOUR CODE HERE
#     # currentWord就是centerword -- a string
#     centerWordIndex = tokens[currentWord]
#     centerword_input_vector = inputVectors[centerWordIndex]

#     for k in range(len(contextWords)):
#         # entryId = tokens[contextWords[k]]
#         word = contextWords[k]
#         entry = db_model.getWordEntry(word)
#         targetWord_output_vector = entry[2]
#         negSamples_output_vectors = db_model.getNegSameples(K, contextWords)

#         # tempCost, tempVgrad, tempUgrad = word2vecCostAndGradient(centerword_input_vector, targetWord_output_vector, negSamples_output_vectors)
#         ___cost, ___cen_grad, ___negSamples_grad = negSampling.get_cost_and_grad(centerword_vector, negSamples_entrys)
#         gradOut += tempUgrad
#         gradIn[centerWordIndex] += tempVgrad
#         cost += tempCost
#     # END YOUR CODE

#     return cost, gradIn, gradOut


def update_o_grad(entry, grad):
    vec = np.array(json.loads(entry[2]))
    zeroArr = np.zeros(int(len(vec) / 2))
    vec_grad = np.concatenate((zeroArr, np.array(grad)), axis=0)
    return vec - vec_grad


def update_i_grad(entry, grad):
    vec = np.array(json.loads(entry[2]))
    zeroArr = np.zeros(int(len(vec) / 2))
    vec_grad = np.concatenate((np.array(grad), zeroArr), axis=0)
    return vec - vec_grad


def skipgram(centerword, contextWords):
    # cost = 0.0
    # windowLength = 3
    # trainingPairs, tokens, wordVectors = wv.getDataset(string, windowLength)
    # gradIn = []
    # gradOut = []
    # for pair in trainingPairs:
    #     centerword = pair[0]
    #     contextWords = pair[1]
    #     sampleNum = 4
    #     centerword_vector = json.loads(db_model.getWordEntrys(centerword)[0][2])
    centerword_entry = db_model.getWordEntrys(centerword)[0]
    centerword_vector = json.loads(centerword_entry[2])

    cost = 0.0
    gradIn = []

    for targetword in contextWords:
        targetword_vector = json.loads(
            db_model.getWordEntrys(targetword)[0][2])
        negSamples_entrys = db_model.getNegSameples(contextWords, sampleNum)
        ___cost, ___cen_grad, ___negSamples_grad = negSampling.get_cost_and_grad(
            centerword_vector, targetword_vector, negSamples_entrys)

        cost += ___cost
        if len(gradIn) == 0:
            gradIn = ___cen_grad
        else:
            gradIn += ___cen_grad

        for index in range(len(negSamples_entrys)):
            curr_entry = negSamples_entrys[index]
            curr_grad = ___negSamples_grad[index]
            print('curr_grad')
            print(curr_grad)

            curr_vec = update_o_grad(curr_entry, curr_grad)
            db_model.update_vec(curr_entry, curr_vec)

        # if len(gradOut) == 0:
        #     gradOut = np.array(___negSamples_grad)
        # else:
        #     gradOut += np.array(___negSamples_grad)
    print('gradIn')
    print(gradIn)
    i_vec = update_i_grad(centerword_entry, gradIn)
    db_model.update_vec(centerword_entry, i_vec)
    print('centerword_entry')

    print(centerword_entry)
    return cost


# db_model.mark_entry_as_treated(entry[0])

entry = db_model.fetch_entry_untreated()
string = entry[2]
# print(string)
windowLength = 3
trainingPairs, tokens, wordVectors = wv.getDataset(string, windowLength)
pair = trainingPairs[0]
centerword = pair[0]
contextWords = pair[1]
sampleNum = 4
centerword_vector = json.loads(db_model.getWordEntrys(centerword)[0][2])


a, b, c = skipgram(centerword, contextWords)
# print(c)
