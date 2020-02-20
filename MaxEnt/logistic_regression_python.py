# coding:utf-8
# Author:codewithzichao
# Date:2020-1-2
# E-mail:lizichao@pku.edu.cn

'''
数据集：Mnist
准确率：0.9919
时间：29.48268699645996
--------------
tips:在加载数据的时候，把>=5为1，<5为0这样处理数据的时候，在同样的训练次数与学习率的时候，最后的准确率只有78%左右。
可能是数据类别太多，导致比较混乱，所以在这里，我采取的是标签为0的为1，不为0的全为0。
这样准确率大大提高了。
'''

import numpy as np
import time


def loadData(fileName):
    '''
    加载数据
    :param fileName:数据路径名
    :return: 特征向量矩阵、还有标签矩阵
    '''
    data_list = [];
    label_list = []

    with open(fileName, "r") as f:
        for line in f.readlines():
            curline = line.strip().split(",")
            if (int(curline[0]) >= 5):
                label_list.append(1)
            else:
                label_list.append(0)
            data_list.append([int(feature) / 255 for feature in curline[1:]])

    data_matrix = np.array(data_list)
    label_matrix = np.array(label_list).reshape(1, -1)
    return data_matrix, label_matrix


def sigmoid(z):
    '''
    定义sigmoid函数
    :param z: 输入
    :return: 返回（0，1）的数
    '''
    result = 1 / (1 + np.exp(-z))
    return result


def initialize_params(feature_dim):
    '''
    初始化参数w,b
    :param feature_dim:实例特征数目
    :return: 参数w,b
    '''
    w = np.zeros((feature_dim, 1))
    b = 0

    return w, b


def propagation(w, b, X, Y):
    '''
    一次前向与反向传播过程
    :param w:参数w
    :param b: 参数b
    :param X: 输入的特征向量
    :param Y: 输入的类别向量
    :return:dw,db,costs
    '''
    N, _ = np.shape(X)  # 训练集数目
    X = X.T
    # print(X.shape)
    A = sigmoid(np.dot(w.T, X) + b)
    # epsilon=1e-5
    cost = -1 / N * (np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)))

    dz = A - Y
    dw = 1 / N * np.dot(X, dz.T)
    db = 1 / N * np.sum(dz)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw, "db": db}

    return grads, cost


def optimization(w, b, X, Y, iterations, learning_rate):
    '''
    优化，使用batch GD
    :param w: 参数w
    :param b: 参数b
    :param X: 输入的特征向量
    :param Y: 输入的类别向量
    :param iterations: 迭代次数(其实就是epoch)
    :param learning_rate: 学习率
    :return: 最优化的参数w,b,以及costs（costs可有可无，取决于你是否想看训练过程中的cost的变化）
    '''
    costs = []

    for iter in range(iterations):
        grads, cost = propagation(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 每100次epoch打印一次信息
        if (iter % 100 == 0):
            costs.append(cost)
            print(f"the current iteration is {iter},the current cost is {cost}.")

        params = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    '''
    预测新实例的类别
    :param w:最优化的参数w
    :param b:最优化的参数b
    :param X:实例的特征向量
    :return:实例的类别
    '''
    N = X.shape[0]
    prediction = np.zeros((1, N))
    X = X.T
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(N):
        if (A[0][i] <= 0.5):
            prediction[0][i] = 0
        else:
            prediction[0][i] = 1

    assert (prediction.shape == (1, N))

    return prediction


def model(train_data, train_label, test_data, test_label, iterations, learning_rate):
    '''
    将上述定义的函数结合起来，就是整个LR模型的执行过程
    :param train_data: 训练数据集
    :param train_label: 训练数据集的标签
    :param test_data: 测试数据集
    :param test_label: 测试数据集的标签
    :param iterations: 迭代次数(epoch)
    :param learning_rate: 学习率
    :return: 在测试数据集上的准确率
    '''
    w, b = initialize_params(train_data.shape[1])
    params, grads, costs = optimization(w, b, train_data, train_label, iterations, learning_rate)

    w = params["w"]
    b = params["b"]

    prediction = predict(w, b, test_data)
    error = 0
    for i in range(prediction.shape[1]):
        if (prediction[0][i] != test_label[0][i]):
            error += 1

    accuracy = (prediction.shape[1] - error) / prediction.shape[1]

    print(f"the accuracy is {accuracy}.")

    d = {"w": w, "b": b, "costs": costs}
    return d


if __name__ == "__main__":
    start = time.time()

    print("start load data.")
    train_data, train_label = loadData("../MnistData/mnist_train.csv")
    test_data, test_label = loadData("../MnistData/mnist_test.csv")
    print("finished load data.")

    d = model(train_data, train_label, test_data, test_label, iterations=200, learning_rate=0.7)

    end = time.time()
    print(f"the total time is {end - start}.")
