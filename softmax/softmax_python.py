# coding:utf-8
# Author:codewithzichao
# Date:2020-1-2
# E-mail:lizichao@pku.edu.cn

'''
数据集：Mnist
准确率：0.1532.
时间：16.96704387664795.
--------------
tips:实现的非常简单，直接将输入数据输入到一个线性层，然后输出10个0-1之间的数，中间没有用任何的隐藏层，
由于是线性变换，所以效果比较差吧，如果加几个隐藏层，效果会好得多！
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
            label_list.append(int(curline[0]))
            data_list.append([int(feature) / 255 for feature in curline[1:]])

    data_matrix = np.array(data_list)
    label_matrix = np.array(label_list).reshape(1, -1)
    return data_matrix, label_matrix


class Softmax:
    def __init__(self, train_data, train_label, iter, learning_rate):
        '''
        构造函数
        :param train_data: 训练数据
        :param train_label: 训练数据的标签类别
        :param iter: epoch的数目
        :param learning_rate: 学习率
        '''
        self.train_data = train_data
        self.train_label = train_label
        self.iter = iter
        self.learning_rate = learning_rate
        self.feature_num = self.train_data.shape[1]
        self.input_num = self.train_data.shape[0]
        self.w, self.b = self.initialize_params(self.feature_num)

    def softmax(self, X):
        '''
        softmax函数
        :param X: 输入数据
        :return: 返回含有10个元素的np.array
        '''
        return np.exp(X) / np.sum(np.exp(X))

    def initialize_params(self, feature_dim):
        '''
        初始化参数w,b
        :param feature_dim:实例特征数目
        :return: 参数w,b
        '''
        w = np.random.uniform(0, 1, (feature_dim, 10))
        b = 0

        return w, b

    def train(self):
        '''
        训练
        :return:返回参数w,b
        '''
        for iter in range(self.iter):
            for i in range(self.input_num):
                x = train_data[i].reshape(-1, 1)
                y = np.zeros((10, 1))
                y[train_label[0][i]] = 1
                y_ = self.softmax(np.dot(self.w.T, x) + self.b)
                self.w -= self.learning_rate * (np.dot((y_ - y), x.T))
                self.b -= self.learning_rate * (y_ - y)
        return self.w, self.b

    def predict(self, digit):
        '''
        预测单个样本的值
        :param digit: 严格样本的特征向量
        :return: 返回预测的样本类别
        '''

        # 返回softmax中概率最大的值
        return np.argmax(np.dot(self.w.T, digit) + self.b)

    def test(self, test_data, test_label):
        '''
        测试
        :param test_data: 测试集的特征向量
        :param test_label: 测试集的标签类别
        :return: 准确率
        '''
        error = 0
        for i in range(test_data.shape[0]):
            if (self.predict(test_data[i]) != test_label[0][i]):
                error += 1
                print(f"the prediction is {self.predict(test_data[i])},the true is {test_label[0][i]}.")

        accuracy = (test_data.shape[0] - error) / test_data.shape[0]

        return accuracy


if __name__ == "__main__":
    start = time.time()

    print("start load data.")
    train_data, train_label = loadData("../MnistData/mnist_train.csv")
    test_data, test_label = loadData("../MnistData/mnist_test.csv")
    print("finished load data.")

    a = Softmax(train_data, train_label, 50, 0.005)
    accuracy = a.test(test_data, test_label)

    end = time.time()
    print(f"the accuracy is {accuracy}.")
    print(f"the total time is {end - start}.")
