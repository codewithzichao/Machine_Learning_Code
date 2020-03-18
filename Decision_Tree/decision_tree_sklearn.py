#coding:utf-8
#Author:codewithzichao
#E-mail:lizichao@pku.edu.cn

'''
数据集：mnist
accuaracy:0.8659.
time:14.435183763504028.
'''

import pandas as pd
import numpy as np
from sklearn import tree
import time

def loadData(fileName):
    #从文件中读取数据
    data=pd.read_csv(fileName,header=None)
    # 将数据从dataframe转化为ndarray
    data=data.values
    #数据第一行为分类结果
    y_label=data[:,0]
    x_label=data[:,1:]
    y_label=np.array(y_label).reshape(-1)
    x_label=np.array(x_label)


    #数据二值化，返回数据
    #因为xi的取值范围为0-255，则计算p(X=xi\Y=y)的时候可能性过多，计算过于繁杂
    # 所以进行二值化
    # y_label为np.ndarray,x_label为np.ndarray

    x_label[x_label<128]=0
    x_label[x_label>=128]=1

    # mp.ndarray
    return x_label,y_label


if __name__=="__main__":
    # 获取当前时间
    start = time.time()

    # 读取训练文件
    print("load train data")
    X_train,y_train = loadData('../MnistData/mnist_train.csv')

    # 读取测试文件
    print('load test data')
    X_test,y_test = loadData('../MnistData/mnist_test.csv')

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train,y_train)

    test_accuracy=clf.score(X_test, y_test)
    print(f"the test_accuracy is {test_accuracy}.")

    end=time.time()

    print(f"the total time is {end-start}.")