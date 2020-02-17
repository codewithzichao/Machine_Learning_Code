# coding=utf-8
# Author:codewithzichao
# Date:2019-12-15
# E-mail:lizichao@pku.edu.cn

'''
数据集：Mnist数据集
模型：感知机模型，对其原始形式与对偶形式均进行了实现
实现方式：python+numpy
结果：
在测试集上的准确率：0.8234
时间：28.92s
'''

import numpy as np
import time


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

    # 将list类型的特征向量，变换成矩阵，维度为（60000，784）
    data_matrix = np.array(np.mat(data_list))
    # 将list类型的标签类别，变换成矩阵，维度为（1,60000）
    label_matrix = np.array(np.mat(label_list))

    return data_matrix, label_matrix


class Perceptron(object):
    '''
    定义Peceptron类，里面包括了其学习算法的原始形式与对偶形式的实现函数
    '''

    def __init__(self, data_matrix, label_matrix, iteration=30, learning_rate=0.0001):
        '''
        构造函数
        :param data_matrix: 数据的特征向量
        :param label_matrix: 数据的标签类别
        :param iteration: 迭代次数,默认为100
        :param learning_rate: 学习率，默认为0.001
        '''
        self.data_matrix = data_matrix
        self.label_matrix = label_matrix
        self.iteration = iteration
        self.learning_rate = learning_rate

    def original_method(self):
        '''
        感知机学习算法的原始形式的实现
        :return: 返回参数w，b
        '''

        data_matrix = self.data_matrix
        label_matrix = self.label_matrix.T
        # input_num表示训练集数目，feature_num表示特征数目
        input_num, feature_num = np.shape(data_matrix)
        print(data_matrix.shape)
        w = np.random.randn(1, feature_num)
        b = np.random.randn()
        # 迭代iteration次
        for iter in range(self.iteration):
            # 在每一个样本上都进行判断
            for i in range(input_num):
                x_i = data_matrix[i]
                y_i = label_matrix[i]
                result = y_i * (np.matmul(w, x_i.T) + b)
                if result <= 0:
                    w = w + self.learning_rate * y_i * x_i
                    b = b + self.learning_rate * y_i
            print(f"this is {iter} round ,the total round is {self.iteration}.")
        assert (w.shape == (1, feature_num))
        return w, b

    def dual_method(self):
        '''
        感知机学习算法的对偶形式的实现
        :return:
        '''
        data_matrix = self.data_matrix
        label_matrix = self.label_matrix.T

        # input_num表示训练集数目，feature_num表示特征数目
        input_num, feature_num = np.shape(data_matrix)
        # 系数a，初始化为全0的(1,input_num)的矩阵
        a = np.zeros((1, input_num))
        # 系数b，初始化为0
        b = 0

        # 计算出gram矩阵
        gram = np.matmul(data_matrix[:, 0:-1], data_matrix[:, 0:-1].T)
        assert (gram.shape == (input_num, input_num))

        # 迭代iteration次
        for iter in range(self.iteration):
            # 在每一个样本上都进行判断
            for i in range(input_num):
                result = 0
                for j in range(input_num):
                    result += a[j] * label_matrix[j] * gram[j, i]
                result += b
                result *= label_matrix[i]

                # 判断当前样本会不会被误分类
                if (result <= 0):
                    a[i] = a[i] + self.learning_rate
                    b = b + self.learning_rate * label_matrix[i]
                else:
                    continue

            print(f"this is {iter} round,the total round is {self.iteration}.")

        w = np.multiply(np.multiply(a, label_matrix.T), data_matrix)
        assert (w.shape == (1, feature_num))

        return w, b


def test_model(test_data_matrix, test_label_matrix, w, b):
    '''
    再测试数据集上测试
    :param test_data_matrix: 测试集的特征向量
    :param test_label_matrix: 测试集的标签类别
    :param w: 计算得到的参数w
    :param b: 计算得到的参数b
    :return: 返回准确率
    '''

    test_input_num, _ = np.shape(test_data_matrix)
    error_num = 0

    # 统计在测试集上数据被误分类的数目
    for i in range(test_input_num):
        result = (test_label_matrix[0, i]) * (np.dot(w, test_data_matrix[i].T) + b)
        if (result <= 0):
            error_num += 1
        else:
            continue

    accuracy = (test_input_num - error_num) / test_input_num
    # 返回模型在测试集上的准确率
    return accuracy


if __name__ == "__main__":
    # 获取当前时间,作为程序开始运行时间
    start = time.time()

    train_data_list, train_label_list = loadData("../MnistData/mnist_train.csv")
    test_data_list, test_label_list = loadData("../MnistData/mnist_test.csv")
    perceptron = Perceptron(train_data_list, train_label_list, iteration=30, learning_rate=0.0001)
    w, b = perceptron.original_method()

    accuracy = test_model(test_data_list, test_label_list, w, b)
    # 获取当前时间，作为程序结束运行时间
    end = time.time()

    # 打印模型在测试集上的准确率
    print(f"accuracy is {accuracy}.")
    # 打印程序运行总时间
    print(f"the total time is {end - start}.")
