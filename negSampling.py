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
    current_vector = np.array(entry['vector'])  # 测试的时候把json取消
    # current_vector = np.array(json.loads(entry[2]))  # 测试的时候把json取消
    halfNum = int(len(current_vector) / 2)
    curr_o_vec = current_vector[halfNum:]
    return curr_o_vec

# def get_cost_and_grad_test(centerword_vector, target_vector, negSamples_list, K=10):

    # assert x.shape == orig_shape
    # assert x.shape == orig_shape

def get_cost_and_grad(centerword_vector, target_vector, negSamples_list, K=10):
    
    assert type(centerword_vector) == np.ndarray
    assert type(negSamples_list) == list
    assert type(negSamples_list[0]) == dict
    assert type(negSamples_list[0]['id']) == int
    assert type(negSamples_list[0]['vector']) == np.ndarray
    assert len(negSamples_list[0]['vector']) == 16


    # _下划线打头的变量为要返回的值
    cen_i_vec, cen_o_vec = divideVec(centerword_vector)
    target_i_vec, target_o_vec = divideVec(target_vector)
    dotProduct = np.sum(target_o_vec * cen_i_vec)
    activation = sigmoid(dotProduct)

    ___cost = - np.log(activation)

    ___cen_i_grad = calcGrad(activation, target_o_vec)
    ___target_o_grad = calcGrad(activation, cen_i_vec)

    ___negSamples_grad = []

    for entry in negSamples_list:
        curr_o_vec = get_o_vec_from_entry(entry)
        dotProduct = np.sum(curr_o_vec * cen_i_vec)
        activation = sigmoid(-dotProduct)

        # 1
        ___cost -= np.log(activation)

        # 2 input word grad, 注意是减去
        ___cen_i_grad -= calcGrad(activation, curr_o_vec)

        # 3 output word grad
        curr_grad = - calcGrad(activation, cen_i_vec)
        
        ___negSamples_grad.append(curr_grad)

    assert type(___cen_i_grad) == np.ndarray
    assert len(___cen_i_grad) == 8
    
    assert type(___target_o_grad) == np.ndarray
    assert len(___target_o_grad) == 8

    assert type(___negSamples_grad) == list
    assert type(___negSamples_grad[0]) == np.ndarray
    assert len(___negSamples_grad[0]) == 8
    
    return ___cost, ___cen_i_grad, ___negSamples_grad,___target_o_grad




