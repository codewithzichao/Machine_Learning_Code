# coding=utf-8
# Author:codewithzichao
# Date:2019-12-20
# E-mail:lizichao@pku.edu.cn


'''
数据集：Mnist(实际只使用了前1000个，当然可以全部使用，如果算力够的话。)
准确率：0.98
时间：0.98
'''

import numpy as np
import time


def loadData(fileName):
    '''
    加载数据
    :param fileName:路径名
    :return: 返回特征向量与标签类别
    在这里需要注意的是：最后使用要np.array，以保证维度。
    当然使用np.mat等也没有问题，但是相关操作容易搞混。
    '''
    data_list = []
    label_list = []

    with open(fileName, "r") as f:
        for line in f.readlines():
            curline = line.strip().split(",")
            if (int(curline[0]) ==0):
                label_list.append(1)
            else:
                label_list.append(-1)
            data_list.append([int(num)/255 for num in curline[1:]])

    data_matrix = np.array(data_list)
    label_matrix = np.array(label_list)
    return data_matrix, label_matrix


class SVM(object):
    def __init__(self, data_matrix, label_matrix, sigma, C, toler, iteration):
        '''
        构造函数
        :param data_matrix: 训练数据集矩阵
        :param label_matrix: 训练数据标签矩阵
        :param sigma: 高斯核函数的sigma
        :param C: 惩罚参数
        :param toler: 松弛变量
        '''

        self.train_data = data_matrix
        self.train_label = label_matrix.T
        self.input_num,self.feature_num = np.shape(self.train_data)

        self.sigma = sigma
        self.C = C
        self.toler = toler
        self.K = self.kernel_for_train()
        self.b = 0
        self.alpha = np.zeros(self.input_num)
        self.E = np.zeros(self.input_num)
        self.iteration = iteration
        self.support_vector = []

    def kernel_for_train(self):
        '''
        计算核函数
        使用的是高斯核 详见“7.3.3 常用核函数” 式7.90
        :return: 高斯核矩阵
        '''
        k=np.zeros((self.input_num,self.input_num))

        for i in range(self.input_num):
            if i % 100 == 0:
                print('construct the kernel:', i, self.input_num)
            X = self.train_data[i, :]
            for j in range(i, self.input_num):
                Z = self.train_data[j, :]
                result =np.matmul((X - Z) , (X - Z).T)
                result = np.exp(-1 * result / (2 * self.sigma**2))
                k[i][j] = result
                k[j][i] = result
        #返回高斯核矩阵
        return k



    def is_satisfied_kkt_condition(self, i):
        '''
        判断alpha_i是否满足KKT条件
        :param i: 索引第i个alpha
        :return: 返回true/false
        '''

        g_xi = self.cal_g_xi(i)
        y_i = self.train_label[i]
        if ((abs(self.alpha[i]) < self.toler) and (y_i * g_xi >= 1)):
            return True
        elif (abs(self.alpha[i] - self.C) < self.toler and (y_i * g_xi <= 1)):
            return True
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) \
                and (abs(y_i * g_xi - 1) < self.toler):
            return True

        return False

    def cal_g_xi(self, i):
        '''
        计算g_xi，从而辅助判断变量是不是满足KKT条件
        :return: 返回g_xi的值,是标量
        '''

        g_xi = 0
        index = [i for i,alpha in enumerate(self.alpha) if alpha!=0]

        for j in index:
            g_xi += self.alpha[j] * self.train_label[j] * self.K[j][i]
        g_xi += self.b

        return g_xi

    def cal_ei(self, i):
        '''
        计算ei的值，标量
        :param i: 表示第i个样本，i=1，2
        :return: 返回Ei
        '''
        return self.cal_g_xi(i) - self.train_label[i]

    def cal_alpha_j(self, e1, i):
        '''
        在SMO中，选择第2个变量

        :param e1: 第一个变量的e1
        :param i: 第一个变量alpha1的下标
        :return: 返回E2，第二个遍历变量的下标
        '''

        e2 = 0
        max_e1_e2 = -1
        max_index = -1

        none_0_e = [i for i, ei in enumerate(self.E) if ei != 0]
        for j in none_0_e:
            # 计算E2
            e2_tmp = self.cal_ei(j)
            # 如果|E1-E2|大于目前最大值
            if abs(e1 - e2_tmp) > max_e1_e2:
                # 更新最大值
                max_e1_e2 = abs(e1 - e2_tmp)
                # 更新最大值E2
                e2 = e2_tmp
                # 更新最大值E2的索引j
                max_index = j
        # 如果列表中没有非0元素了（对应程序最开始运行时的情况）
        if max_index == -1:
            max_index = i
            while max_index == i:
                # 获得随机数，如果随机数与第一个变量的下标i一致则重新随机
                max_index = int(np.random.uniform(0, self.input_num))
            # 获得E2
            e2 = self.cal_ei(max_index)

        return e2, max_index

    def train(self):
        cur_iter_step = 0
        params_changed = 1
        #结束循环条件：1.达到循环次数；2.当参数不再发生变化的时候，说明收敛了
        while ((cur_iter_step < self.iteration) and (params_changed > 0)):
            print(f"the current iteration is {cur_iter_step},the total iteration is {self.iteration}.")
            cur_iter_step += 1
            params_changed = 0

            for i in range(self.input_num):
                if (self.is_satisfied_kkt_condition(i) == False):
                    e1 = self.cal_ei(i)
                    e2, j = self.cal_alpha_j(e1, i)

                    y1 = self.train_label[i]
                    y2 = self.train_label[j]

                    alpha_old_1 = self.alpha[i]
                    alpha_old_2 = self.alpha[j]

                    if (y1 != y2):
                        L = max(0, alpha_old_2 - alpha_old_1)
                        H = min(self.C, self.C - alpha_old_1 + alpha_old_2)
                    else:
                        L = max(0, alpha_old_1 + alpha_old_2 - self.C)
                        H = min(self.C, alpha_old_1 + alpha_old_2)
                    if (L == H):
                        continue

                    k11 = self.K[1][1]
                    k22 = self.K[2][2]
                    k12 = self.K[1][2]
                    k21 = self.K[2][1]

                    alpha_new_2 = alpha_old_2 + y2 * (e1 - e2) / (k11 + k22 - 2 * k12)
                    if (alpha_new_2 < L):
                        alpha_new_2 = L
                    elif (alpha_new_2 > H):
                        alpha_new_2 = H

                    alpha_new_1 = alpha_old_1 + y1 * y2 * (alpha_old_2 - alpha_new_2)

                    b1_new = -1 * e1 - y1 * k11 * (alpha_new_1 - alpha_old_1) \
                             - y2 * k21 * (alpha_new_2 - alpha_old_2) + self.b
                    b2_new = -1 * e2 - y1 * k12 * (alpha_new_1 - alpha_old_1) \
                             - y2 * k22 * (alpha_new_2 - alpha_old_2) + self.b

                    if (alpha_new_1 > 0) and (alpha_new_1 < self.C):
                        b_new = b1_new
                    elif (alpha_new_2 > 0) and (alpha_new_2 < self.C):
                        b_new = b2_new
                    else:
                        b_new = (b1_new + b2_new) / 2

                    self.alpha[i] = alpha_new_1
                    self.alpha[j] = alpha_new_2
                    self.b = b_new

                    self.E[i] = self.cal_ei(i)
                    self.E[j] = self.cal_ei(j)

                    if abs(alpha_new_2 - alpha_old_2) >= 0.00001:
                        params_changed += 1

        for i in range(self.input_num):
            # 如果α>0，说明是支持向量
            if self.alpha[i] > 0:
                # 将支持向量的索引保存起来
                self.support_vector.append(i)

    def kernel_for_predict(self, x, z):
        '''
        计算核函数，用于测试集与开发集
        :param x:向量x
        :param z: 向量z
        :return: 返回核函数
        '''

        result = np.matmul((x - z), (x - z).T)
        result = np.exp(-result / (2 * self.sigma ** 2))

        return result

    def predict(self, x):
        '''
        预测单个样本的类别
        :param x: 样本的特征向量
        :return: 样本的预测类别
        '''

        result = 0
        for i in self.support_vector:
            result_temp = self.kernel_for_predict(x, self.train_data[i])
            result += self.alpha[i] * self.train_label[i] * result_temp

        result += self.b
        return np.sign(result)

    def test(self, test_data, test_label):
        '''
        在测试集上测试模型
        :param test_train:测试集的特征向量
        :param test_label:测试集的类别
        :return:准确率
        '''
        test_label=test_label.T
        error = 0
        test_input_num = test_data.shape[0]

        for i in range(test_input_num):
            result = self.predict(test_data[i])

            if (result != test_label[i]):
                error += 1
            else:
                continue

        return (test_input_num - error) / test_input_num


if __name__ == "__main__":
    start = time.time()

    train_data, train_label = loadData("../MnistData/mnist_train.csv")
    test_data, test_label = loadData("../MnistData/mnist_test.csv")
    print("finished load data.")
    svm = SVM(train_data[:1000], train_label[:1000], sigma=10, C=200, toler=0.001,iteration=50)
    svm.train()
    print("finish the training.")
    accuracy = svm.test(test_data[:100], test_label[:100])
    print(f"accuracy is {accuracy}.")

    end = time.time()

    print(f"the total time is {end - start}.")
