#coding:utf-8
#Author:codewithzichao
#E-mail:lizichao@pku.edu.cn

'''
数据集：Mnist数据集(只使用了1000来训练，只使用了1000来测试。)
结果(准确率)：0.799
时间：16.832828998565674
---------------------------
果然，自己写的python没有编写kdtree等部分，效果与时间上都比不上sklearn。
'''

import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier

def loadData(fileName):
    '''
    加载数据
    :param fileName: 数据路径
    :return: 返回特征向量与标签类别
    '''
    data_list=[]
    label_list=[]

    with open(fileName,"r") as f:
        for line in f.readlines():
            curline=line.strip().split(",")

            data_list.append([int(feature) for feature in curline[1:]])
            label_list.append(int(curline[0]))

        data_matrix=np.array(data_list)
        label_matrix=np.array(label_list)

        return data_matrix,label_matrix


if __name__=="__main__":
    start = time.time()

    print("start load data.")
    train_data, train_label = loadData("../MnistData/mnist_train.csv")
    test_data, test_label = loadData("../MnistData/mnist_test.csv")
    print("finished load data.")

    knn=KNeighborsClassifier(n_neighbors=10)
    knn.fit(train_data[:1000],train_label[:1000])

    prediction=knn.predict(test_data[:1000])
    for i in range(1000):
        print(f"predict is {prediction[i]},the true is {test_label[i]}.")
    accuracy=knn.score(test_data[:1000],test_label[:1000])
    print(f"the accuracy is {accuracy}.")

    end=time.time()

    print(f"the total time is {end-start}.")

