# coding:utf-8
# Author:codewithzichao
# E-mail:lzichao@pku.edu.cn

'''
hmmlearn中有两个模型：高斯HMM与多项式HMM，分别对应于：变量是连续的与离散的。

'''
import numpy as np
from hmmlearn import hmm

# 3 个盒子状态
states = ['box1', 'box2', 'box3']
# 2 个球观察状态
observations = ['red', 'white']
# 初始化概率
start_probability = np.array([0.2, 0.4, 0.4])
# 转移概率
transition_probability = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])

# 发射状态概率
emission_probability = np.array([
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
])

if __name__ == "__main__":
    # 建立模型，设置参数
    model = hmm.MultinomialHMM(n_components=len(states))
    model.startprob_ = start_probability
    model.transmat_ = transition_probability
    model.emissionprob_ = emission_probability

    # 预测问题,执行Viterbi算法
    seen = np.array([0, 1, 0])
    logprob, box = model.decode(seen.reshape(-1, 1), algorithm='viterbi')
    print('The ball picked:', ','.join(map(lambda x: observations[x], seen)))
    print('The hidden box:', ','.join(map(lambda x: states[x], box)))

    box_pre = model.predict(seen.reshape(-1, 1))
    print('The ball picked:', ','.join(map(lambda x: observations[x], seen)))
    print('The hidden box predict:', ','.join(map(lambda x: states[x], box_pre)))
    # 观测序列的概率计算问题
    # score函数返回的是以自然对数为底的对数概率值
    # ln0.13022≈−2.0385
    print(model.score(seen.reshape(-1, 1)))

    print("-------------------------------")

    # 学习问题
    states = ["box1", "box2", "box3"]
    n_states = len(states)

    observations = ["red", "white"]
    n_observations = len(observations)

    model = hmm.MultinomialHMM(n_components=n_states)
    # 三个观测序列，用来训练
    X = np.array([[0, 1, 0, 1], [0, 0, 0, 1], [1, 0, 1, 1]])
    model.fit(X)  # 训练模型
    print(model.startprob_)  # 得到初始概率矩阵
    print(model.transmat_)  # 得到状态转移矩阵
    print(model.emissionprob_)  # 得到观测概率分布
    # 概率计算
    print(model.score(X))
