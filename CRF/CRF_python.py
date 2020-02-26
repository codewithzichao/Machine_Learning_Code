#coding:utf-8
#Author:codewithzichao
#E-mail:lizichao@pku.edu.cn

'''
由于CRF模型，一般都是来解决其Decoding问题，所以在此，我只实现Viterbi算法～
'''

import numpy as np


class CRF(object):
    '''实现条件随机场预测问题的维特比算法
    '''

    def __init__(self, V, VW, E, EW):
        '''
        :param V:是定义在节点上的特征函数，称为状态特征
        :param VW:是V对应的权值
        :param E:是定义在边上的特征函数，称为转移特征
        :param EW:是E对应的权值
        '''
        self.V = V  # 点分布表
        self.VW = VW  # 点权值表
        self.E = E  # 边分布表
        self.EW = EW  # 边权值表
        self.D = []  # Delta表，最大非规范化概率的局部状态路径概率
        self.P = []  # Psi表，当前状态和最优前导状态的索引表s
        self.BP = []  # BestPath，最优路径
        return

    def Viterbi(self):
        '''
        条件随机场预测问题的维特比算法，此算法一定要结合CRF参数化形式对应的状态路径图来理解，更容易理解.
        '''
        self.D = np.full(shape=(np.shape(self.V)), fill_value=.0)
        self.P = np.full(shape=(np.shape(self.V)), fill_value=.0)
        for i in range(np.shape(self.V)[0]):
            # 初始化
            if 0 == i:
                self.D[i] = np.multiply(self.V[i], self.VW[i])
                self.P[i] = np.array([0, 0])
                print('self.V[%d]=' % i, self.V[i], 'self.VW[%d]=' % i, self.VW[i], 'self.D[%d]=' % i, self.D[i])
                print('self.P:', self.P)
                pass
            # 递推求解布局最优状态路径
            else:
                for y in range(np.shape(self.V)[1]):  # delta[i][y=1,2...]
                    for l in range(np.shape(self.V)[1]):  # V[i-1][l=1,2...]
                        delta = 0.0
                        delta += self.D[i - 1, l]  # 前导状态的最优状态路径的概率
                        delta += self.E[i - 1][l, y] * self.EW[i - 1][l, y]  # 前导状态到当前状体的转移概率
                        delta += self.V[i, y] * self.VW[i, y]  # 当前状态的概率
                        print('(x%d,y=%d)-->(x%d,y=%d):%.2f + %.2f + %.2f=' % (i - 1, l, i, y, \
                                                                               self.D[i - 1, l], \
                                                                               self.E[i - 1][l, y] * self.EW[i - 1][
                                                                                   l, y], \
                                                                               self.V[i, y] * self.VW[i, y]), delta)
                        if 0 == l or delta > self.D[i, y]:
                            self.D[i, y] = delta
                            self.P[i, y] = l
                    print('self.D[x%d,y=%d]=%.2f\n' % (i, y, self.D[i, y]))
        print('self.Delta:\n', self.D)
        print('self.Psi:\n', self.P)

        # 返回，得到所有的最优前导状态
        N = np.shape(self.V)[0]
        self.BP = np.full(shape=(N,), fill_value=0.0)
        t_range = -1 * np.array(sorted(-1 * np.arange(N)))
        for t in t_range:
            if N - 1 == t:  # 得到最优状态
                self.BP[t] = np.argmax(self.D[-1])
            else:  # 得到最优前导状态
                self.BP[t] = self.P[t + 1, int(self.BP[t + 1])]

        # 最优状态路径表现在存储的是状态的下标，我们执行存储值+1转换成示例中的状态值
        # 也可以不用转换，只要你能理解，self.BP中存储的0是状态1就可以~~~~
        self.BP += 1

        print('最优状态路径为：', self.BP)
        return self.BP



if __name__ == '__main__':
    S = np.array([[1, 1],  # X1:S(Y1=1), S(Y1=2)
                  [1, 1],  # X2:S(Y2=1), S(Y2=2)
                  [1, 1]])  # X3:S(Y3=1), S(Y3=1)
    SW = np.array([[1.0, 0.5],  # X1:SW(Y1=1), SW(Y1=2)
                   [0.8, 0.5],  # X2:SW(Y2=1), SW(Y2=2)
                   [0.8, 0.5]])  # X3:SW(Y3=1), SW(Y3=1)
    E = np.array([[[1, 1],  # Edge:Y1=1--->(Y2=1, Y2=2)
                   [1, 0]],  # Edge:Y1=2--->(Y2=1, Y2=2)
                  [[0, 1],  # Edge:Y2=1--->(Y3=1, Y3=2)
                   [1, 1]]])  # Edge:Y2=2--->(Y3=1, Y3=2)
    EW = np.array([[[0.6, 1],  # EdgeW:Y1=1--->(Y2=1, Y2=2)
                    [1, 0.0]],  # EdgeW:Y1=2--->(Y2=1, Y2=2)
                   [[0.0, 1],  # EdgeW:Y2=1--->(Y3=1, Y3=2)
                    [1, 0.2]]])  # EdgeW:Y2=2--->(Y3=1, Y3=2)

    crf = CRF(S, SW, E, EW)
    ret = crf.Viterbi()
    print('最优状态路径为:', ret)
