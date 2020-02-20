# coding:utf-8
# Author:codewithzichao
# Date:2020-1-2
# E-mail:lizichao@pku.edu.cn

'''
数据集：Mnist
准确率：0.8707.
时间：89.82440423965454.
'''

import numpy as np
import time

from sklearn import linear_model
from sklearn.model_selection import train_test_split


def loadData(fileName):
    '''
    加载数据
    :param fileName:数据路径名
    :return: 特征向量矩阵、还有标签矩阵
    '''
    data_list = []
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
    label_matrix = np.array(label_list)
    return data_matrix, label_matrix


if __name__ == "__main__":
    start = time.time()

    print("start load data.")
    train_data, train_label = loadData("../Mnistdata/mnist_train.csv")
    test_data, test_label = loadData("../MnistData/mnist_test.csv")
    print("finished load data.")

    # 默认迭代次数为100,使用的算法是lbfgs，使用L2正则化。这里要加大迭代次数，要不然的化，不会收敛。
    clf = linear_model.LogisticRegression(max_iter=1000)
    clf.fit(train_data, train_label)

    accuracy = clf.score(test_data, test_label)
    print(f"the  accuracy is {accuracy}.")

    end = time.time()
    print(f"the total time is {end - start}.")
