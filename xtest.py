#!/usr/bin/env python
import numpy as np
import random
from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def test_word2vec():    
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