#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


"""
global notice:
    C -- integer, context size

"""

# 把不同行的，放在同一量纲下
def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    # 每一项除以欧几里得距离，这是什么标准化？
    transposeX =  x.T
    x = (transposeX / np.sqrt(np.sum(transposeX**2, axis=0))).T
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print(x)
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE

    # (5,)    
    outputLayer = np.dot(outputVectors,predicted)

    activedOutputLayer = softmax(outputLayer)

    cost = - np.log(activedOutputLayer[target])

    # print(np.sum(activedOutputLayer.reshape(-1,1) * outputVectors,axis = 0))
    # print(outputVectors[target])

    columnVector = activedOutputLayer.reshape(-1,1) 
    gradPred = - outputVectors[target] + np.sum(columnVector*outputVectors,axis = 0)

    onehot = np.zeros(outputVectors.shape[0])
    onehot[target] = 1
    y_Minus_y = activedOutputLayer - onehot
    grad = predicted*y_Minus_y.reshape(-1,1)
    ### END YOUR CODE
    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """
    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
    # 选择了K个sample，再加上target K+1个
    # sample可以重复的

    ### YOUR CODE HERE
    # 选中的词 与 中心词 相乘：
    uTv = np.sum(outputVectors[target]*predicted)
    cost =  - np.log(sigmoid(uTv))
    
    partGradsOfVc = np.zeros(outputVectors.shape[1])
    grad = np.zeros(outputVectors.shape)

    for ind in range(len(indices)):
        if ind != 0:
            ukTv = np.sum(outputVectors[indices[ind]]*predicted)
            cost = cost - np.log(sigmoid(-ukTv))
            # 计算predict的grad
            partGradsOfVc = partGradsOfVc + (sigmoid(-ukTv) - 1)*outputVectors[indices[ind]]
            # 求 Uk 的grad【 重 复 的 也 要 算 进 去 ！ ！ ！】
            grad[indices[ind]] =  grad[indices[ind]] - (sigmoid(-ukTv) - 1)*predicted 
    # 减去循环的部分
    gradPred = (sigmoid(uTv) - 1)*outputVectors[target] - partGradsOfVc
    # 求 Uo 的grad
    grad[target] = (sigmoid(uTv) - 1)*predicted

    ### END YOUR CODE
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    target = tokens[currentWord]
    predicted = inputVectors[target]

    for k in range(len(contextWords)):
        tempExpect = tokens[contextWords[k]]
        tempCost, tempVgrad, tempUgrad = word2vecCostAndGradient(predicted, tempExpect, outputVectors, dataset)
        gradOut += tempUgrad
        gradIn[target] += tempVgrad
        cost += tempCost
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

"""
     wrapper是用来干嘛的
     wordVectors是什么
     C是什么 
        从1到C随机取一个整数
     
"""
def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 1
    cost = 0.0

    # wordVectors 10x3的正态分布, normalize过
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0] # vacabulary的总length
    # 这里为什么是 N/2, 随机启动的意思嘛
    inputVectors = wordVectors[:int(N/2),:] #前5个
    outputVectors = wordVectors[int(N/2):,:] #后5个
    for i in range(batchsize): 
        C1 = random.randint(1,C) # 随机窗口大小
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        # 这里又跳到skip gram model,这么多参数能不能行，日了狗 
        # tokens -- a dictionary that maps words to their indices in
        #          the word vector list

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        
        cost += c / batchsize / denom
        grad[:int(N/2), :] += gin / batchsize / denom
        grad[int(N/2):, :] += gout / batchsize / denom
    return cost, grad


def test_word2vec():
    
    """ Interface to the dataset for negative sampling """ 
    # 声明一个类， 所以说这个是“接口”
    dataset = type('dummy', (), {})() 
    
    # 0到4中间随机选一个
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    """
    返回两个
        1.一个字符，a到e中的一个
        2.返回一个list，长度是输入数值的2倍
    """
    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    # 第一个方法有什么用
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)

    # note:这个是正态分布
    # 还是不知道这个normalizeRows有什么意义，用在正态分布上并无太大意义
    dummy_vectors = normalizeRows(np.random.randn(10,3))

    # k v 对象
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)


    print("\n ==== Gradient check for CBOW      ====")
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
    #     dummy_vectors)
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
    #     dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))
    # print(cbow("a", 2, ["a", "b", "c", "a"],
    #     dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    # print(cbow("a", 2, ["a", "b", "a", "c"],
    #     dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
    #     negSamplingCostAndGradient))

if __name__ == "__main__":
    # test_normalize_rows()
    test_word2vec()