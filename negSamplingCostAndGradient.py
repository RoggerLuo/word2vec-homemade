#!/usr/bin/env python
import numpy as np
import random
import json
from q2_sigmoid import sigmoid, sigmoid_grad


def divideVec(vec):
    N = len(vec)
    a = np.array(vec[:int(N / 2)])  # 前一半
    b = np.array(vec[int(N / 2):])  # 后一半
    return a, b


def calcGrad(activation, vector):
    deviation = activation - 1
    grad = deviation * vector
    return grad


def get_o_vec_from_entry(entry):
    current_vector = np.array(json.loads(entry[2])) #测试的时候把json取消
    halfNum = int(len(current_vector) / 2)
    curr_o_vec = current_vector[halfNum:]
    return curr_o_vec


def negSamplingCostAndGradient(centerword_vector, negSamples_entrys, K=10):
    # _下划线打头的变量为要返回的值
    cen_i_vec, cen_o_vec = divideVec(centerword_vector)

    dotProduct = np.sum(cen_o_vec * cen_i_vec)
    activation = sigmoid(dotProduct)

    _cost = - np.log(activation)

    cen_i_grad = calcGrad(activation, cen_o_vec)
    cen_o_grad = calcGrad(activation, cen_i_vec)

    _negSamples_grad = []
    for entry in negSamples_entrys:
        curr_o_vec = get_o_vec_from_entry(entry)

        dotProduct = np.sum(curr_o_vec * cen_i_vec)
        activation = sigmoid(-dotProduct)

        _cost -= np.log(activation)

        # input word grad, 注意是减去
        cen_i_grad -= calcGrad(activation, curr_o_vec)

        # output word grad
        curr_grad = - calcGrad(activation, cen_i_vec)
        curr_grad = curr_grad.tolist()
        _negSamples_grad.append(curr_grad)

    _cen_grad = np.concatenate((cen_i_grad, cen_o_grad), axis=0)
    return _cost, _cen_grad, _negSamples_grad
