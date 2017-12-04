#!/usr/bin/env python
import numpy as np
import random
import db_model
import negSampling
from q2_sigmoid import sigmoid, sigmoid_grad

def negSamplingCostAndGradient_original():
    predicted = [0.02732849, 0.0267334, 0.00875092, -0.02406311, 0.02145386, 0.02870178, -0.02508545, -0.01371002]
    predicted_output = [0.02059937, 0.02067566, 0.01629639, 0.01783752, -0.02519226, 0.02288818, 0.02467346, -0.00189781]
    outputVectors = [
        [0.02059937, 0.02067566, 0.01629639, 0.01783752, -0.02519226, 0.02288818, 0.02467346, -0.00189781], #这是target output
        [-0.0038166, 0.01318359, -0.03045654, 0.01508331, -0.0158844, -0.01672363, -0.02540588, -0.02848816],
        [-0.00096607, -0.01554108, -0.0136261, -0.00453568, 0.02386475, -0.01512146, -0.00540161, -0.01754761],
        [-0.01858521, -0.02912903, 0.01557159, -0.01390076, -0.02276611, -0.02481079, -0.00237083, 0.02049255]
    ]
    
    predicted = np.array(predicted)
    predicted_output = np.array(predicted_output)
    outputVectors = np.array(outputVectors)
    indices = [0,1,2,3]

    ### YOUR CODE HERE
    # 选中的词 与 中心词 相乘：
    uTv = np.sum(predicted_output * predicted)
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
    gradPred = (sigmoid(uTv) - 1)*predicted_output - partGradsOfVc
    # 求 Uo 的grad
    UoGrad = (sigmoid(uTv) - 1)*predicted

    ### END YOUR CODE
    return cost, gradPred, grad,UoGrad
a,b,c,d = negSamplingCostAndGradient_original()


# 两个拼起来就是下面这个
centerword_vector = [0.02732849, 0.0267334, 0.00875092, -0.02406311, 0.02145386, 0.02870178, -0.02508545, -0.01371002,       -100.02633667, -99.00881195, -0.01450348, 0.00855255, -0.01534271, 0.01309204, -0.00343513, 0.02812195]
centerword_vector = np.array(centerword_vector)
tw = [-220.02633667, -229.00881195, -0.01450348, 0.00855255, -0.01534271, 0.01309204, -0.00343513, 0.02812195,         0.02059937, 0.02067566, 0.01629639, 0.01783752, -0.02519226, 0.02288818, 0.02467346, -0.00189781]
tw = np.array(tw)

negSamples_entrys = [
    {'id':111,'vector':np.array([0.013374, -0.020279, -0.022751, -0.024963, 0.029602, 0.020157, 0.023529, -0.001051,     -0.0038166, 0.01318359, -0.03045654, 0.01508331, -0.0158844, -0.01672363, -0.02540588, -0.02848816]) },
    {'id':112,'vector':np.array([0.01738, 0.011101, 0.017105, -0.007374, 0.016891, 0.002399, 0.001511, -0.029419,      -0.00096607, -0.01554108, -0.0136261, -0.00453568, 0.02386475, -0.01512146, -0.00540161, -0.01754761]) },
    {'id':113,'vector':np.array([-0.03006, -0.008041, -0.002583, -0.010201, -0.00684, -0.016983, 0.008804, 0.017731,     -0.01858521, -0.02912903, 0.01557159, -0.01390076, -0.02276611, -0.02481079, -0.00237083, 0.02049255]) }
]

a2, b2, c2, d2 = negSampling.get_cost_and_grad(centerword_vector,tw, negSamples_entrys)

print('------结果对比------------------------')
print(a)
print(a2)
print('-----')
print(b)
print(b2)
print('------')
print(c)
print(c2)
print('------')
print(d)
print(d2)
print('------------------------------')
