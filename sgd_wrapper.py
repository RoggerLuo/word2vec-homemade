 #!/usr/bin/env python
import numpy as np
import random
from skipgram import skipgram
from getContext import getTrainingPairs_of_one_note

# from q1_softmax import softmax
# from q2_gradcheck import gradcheck_naive
# from q2_sigmoid import sigmoid, sigmoid_grad
def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient):
    if word2vecModel == skipgram:
        denom = 1
    else:
        denom = 1
    # batchsize = 20
    cost = 0.0

    # wordeVectors是把input和output都拼在了一起
    grad = np.zeros(wordVectors.shape)
    # inputVectors = wordVectors[:int(N/2),:] #前一半
    # outputVectors = wordVectors[int(N/2):,:] #后一半
    
    # N = wordVectors.shape[0] # vacabulary的总length
    # inputVectors = wordVectors[:,:int(N/2)] #前一半
    # outputVectors = wordVectors[:,int(N/2):] #后一半
    
    #准备要training的 word pairs
    windowLength = 5
    pairGroup = getTrainingPairs_of_one_note(windowLength) # 从文章中生成pair
    
    #准备word vec，然后分割成input和output

    #准备好token

    #训练完以后保存new vec

    batchsize = len(pairGroup) 
    for i in range(batchsize): 
        centerword, context = pairGroup[i] 

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        
        # 拼回去
        cost += c / batchsize / denom
        grad[:int(N/2), :] += gin / batchsize / denom  # 把center word (input)的grad加上去
        grad[int(N/2):, :] += gout / batchsize / denom # 把output vectoer update

    return cost, grad
