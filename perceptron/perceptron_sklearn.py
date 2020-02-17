# coding=utf-8
# Author:codewithzichao
# Date:2019-12-15
# E-mail:lizichao@pku.edu.cn

'''
数据集：Mnist数据集
模型：感知机模型，对其原始形式与对偶形式均进行了实现
实现方式：使用scikit-learn库
结果：
在测试集上的准确率：0.7849
时间：25,72s
'''

import numpy as np
import time
from sklearn.linear_model import Perceptron


def loadData(fileName):
    '''
    从fileName数据文件中加载Mnist数据集
    :param fileName: 数据集的路径
    :return: 返回数据的特征向量与标签类别
    '''
    # 存放数据的特征向量
    data_list = []
    # 存放数据的标签类别
    label_list = []

    # 读取文件，将特征向量与标签分别存入data_list与label_list
    with open(fileName, "r") as f:
        for line in f.readlines():
            curline = line.strip().split(",")
            data_list.append([int(feature) for feature in curline[1:]])
            if int(curline[0]) >= 5:
                label_list.append(1)
            else:
                label_list.append(-1)

    data_matrix = np.array(np.mat(data_list))
    label_matrix = np.array(np.mat(label_list))
    return data_matrix, label_matrix


if __name__ == "__main__":
    start = time.time()

    # 定义感知机
    # n_iter_no_change表示迭代次数，eta0表示学习率，shuffle表示是否打乱数据集
    clf = Perceptron(n_iter_no_change=30, eta0=0.0001, shuffle=False)
    # 使用训练数据进行训练
    train_data_matrix, train_label_matrix = loadData("../MnistData/mnist_train.csv")
    test_data_matrix, test_label_matrix = loadData("../MnistData/mnist_test.csv")

    print(train_data_matrix.shape)
    print(test_data_matrix.shape)

    train_label_matrix = np.squeeze(train_label_matrix)
    test_label_matrix = np.squeeze(test_label_matrix)

    print(train_label_matrix.shape)
    print(test_label_matrix.shape)

    # 训练模型
    clf.fit(train_data_matrix, train_label_matrix)

    # 利用测试集进行验证，得到模型在测试集上的准确率
    accuracy = clf.score(test_data_matrix, test_label_matrix)

    end = time.time()
    print(f"accuracy is {accuracy}.")
    print(f"the total time is {end - start}.")
