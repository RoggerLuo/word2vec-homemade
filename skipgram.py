#!/usr/bin/env python
import numpy as np
import random
import db_model
from negSamplingCostAndGradient import negSamplingCostAndGradient
from q2_sigmoid import sigmoid, sigmoid_grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient):
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    centerWordIndex = tokens[currentWord] # currentWord就是centerword -- a string 
    centerword_input_vector = inputVectors[centerWordIndex]

    for k in range(len(contextWords)):
        # entryId = tokens[contextWords[k]]
        word = contextWords[k]
        entry = db_model.getWordEntry(word)
        targetWord_output_vector = entry[2]
        negSamples_output_vectors = db_model.getNegSameples(K,contextWords)

        tempCost, tempVgrad, tempUgrad = word2vecCostAndGradient(centerword_input_vector, targetWord_output_vector, negSamples_output_vectors)
        gradOut += tempUgrad
        gradIn[centerWordIndex] += tempVgrad
        cost += tempCost
    ### END YOUR CODE

    return cost, gradIn, gradOut




