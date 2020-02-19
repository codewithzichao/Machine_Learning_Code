# coding=utf-8
# Author:codewithzichao
# E-mail:lizichao@pku.edu.cn
# Date:2019-12-30

'''
数据集：Mnist
准确率：0.8433
时间：130.05937218666077
'''

import numpy  as np
import time


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


class Naive_Bayes(object):
    def __init__(self, train_data, train_label):
        '''
        构造函数
        :param train_data:训练集的特征向量
        :param train_label: 训练集的类别标签
        '''
        self.train_data = train_data
        self.train_label = train_label
        self.input_num, self.feature_num = self.train_data.shape  # input_num、feature_num表示训练集数目、特征数目
        self.classes_num = self.count_classes()
        self.p_y, self.p_x_y = self.get_probabilities()

    def count_classes(self):
        '''
        计算类别数目
        :return:类别数
        '''
        s = set()
        for i in self.train_label:
            if i not in s:
                s.add(i)
        return len(s)

    def get_probabilities(self):
        '''
        计算先验概率与条件概率
        :return: 返回先验概率与条件概率
        '''
        print("start training")
        p_y = np.zeros(self.classes_num)
        p_x_y = np.zeros((self.classes_num, self.input_num, 2))

        # 计算先验概率p_y
        for i in range(self.classes_num):
            p_y[i] = (np.sum((self.train_label == i)) + 1) / (self.input_num + self.classes_num)
        p_y = np.log(p_y)

        # 计算条件概率
        for i in range(self.input_num):
            label = self.train_label[i]
            x = self.train_data[i]
            for j in range(self.feature_num):
                p_x_y[label][j][x[j]] += 1

        for i in range(self.classes_num):
            for j in range(self.feature_num):
                p_x_y_0 = p_x_y[i][j][0]
                p_x_y_1 = p_x_y[i][j][1]

                p_x_y[i][j][0] = np.log((p_x_y_0 + 1) / (p_x_y_0 + p_x_y_1 + 2))
                p_x_y[i][j][1] = np.log((p_x_y_1 + 1) / (p_x_y_0 + p_x_y_1 + 2))
        print("finished training.")
        return p_y, p_x_y

    def naive_bayes_predict(self, x):
        '''
        预测单个实例x的类别标签
        :param x: 特征向量
        :return: x的类别标签
        '''

        p = np.zeros(self.classes_num)
        p_y, p_x_y = self.p_y, self.p_x_y

        for i in range(self.classes_num):
            for j in range(self.feature_num):
                p[i] += p_x_y[i][j][x[j]]
            p[i] += p_y[i]

        return np.argmax(p)

    def test_model(self, test_train, test_label):
        '''
        在整个测试集上测试模型
        :param test_train: 测试集的特征向量
        :param test_label: 测试集的类别标签
        :return: 准确率
        '''
        print("start testing")
        error = 0
        for i in range(len(test_label)):
            if (self.naive_bayes_predict(test_train[i]) != test_label[i]):
                error += 1
            else:
                continue

        accuarcy = (len(test_label) - error) / (len(test_label))
        print("finished testing.")
        return accuarcy


if __name__ == "__main__":
    start = time.time()

    print("start loading data.")
    train_data, train_label = loadData("../MnistData/mnist_train.csv")
    test_data, test_label = loadData("../MnistData/mnist_test.csv")
    print("finished load data.")

    a = Naive_Bayes(train_data, train_label)
    accuracy = a.test_model(test_data, test_label)
    print(f"the accuracy is {accuracy}.")

    end = time.time()
    print(f"the total time is {end - start}.")
