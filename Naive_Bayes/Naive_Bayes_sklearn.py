# coding=utf-8
# Author:codewithzichao
# E-mail:lizichao@pku.edu.cn
# Date:2019-12-30

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def loadData(fileName):
    data_list = []
    label_list = []
    with open(fileName, "r") as f:
        for line in f.readlines():
            curline = line.strip().split(",")
            label_list.append(int(curline[0]))
            data_list.append([int(int(feature) > 128) for feature in curline[1:]])  # 二值化，保证每一个特征只能取到0和1两个值

    data_matrix = np.array(data_list)
    label_matrix = np.array(label_list)

    return data_matrix, label_matrix

if __name__=="__main__":

    print("start loading data.")
    train_data, train_label = loadData("../MnistData/mnist_train.csv")
    test_data, test_label = loadData("../MnistData/mnist_test.csv")
    print("finished load data.")

    clf=MultinomialNB()
    clf.fit(train_data,train_label)

    accuracy=clf.score(test_data,test_label)

    print(f"the accuracy is {accuracy}.")

    test_predict=clf.predict(test_data)
    print(classification_report(test_label, test_predict))
