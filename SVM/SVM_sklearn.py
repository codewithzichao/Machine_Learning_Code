# coding=utf-8
# Author:codewithzichao
# Date:2019-12-20
# E-mail:lizichao@pku.edu.cn

import numpy as np
import time
from sklearn import svm


def loadData(fileName):
    data_list = []
    label_list = []

    with open(fileName, "r") as f:
        for line in f.readlines():
            curline = line.strip().split(",")
            if (int(curline[0]) == 0):
                label_list.append(1)
            else:
                label_list.append(-1)
            data_list.append([int(feature) for feature in curline[1:]])

    data_matrix = np.array(data_list)
    label_matrix = np.array(label_list)
    return data_matrix, label_matrix


if __name__ == "__main__":
    train_data, train_label = loadData("../MnistData/mnist_train.csv")
    test_data, test_label = loadData("../Mnistdata/mnist_test.csv")
    print("finished load data.")
    #创建模型
    clf = svm.SVC()
    #训练模型
    clf.fit(train_data[:1000], train_label[:1000])
    print("finished training.")
    #在测试集上测试模型
    accuracy = clf.score(test_data, test_label)
    print(f"the accuracy is {accuracy}.")
