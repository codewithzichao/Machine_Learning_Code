#coding:utf-8
#Author:codewithzichao
#E-mail:lizichao@pku.edu.cn

'''
数据集：Mnist数据集(只使用了1000来训练，只使用了1000来测试。)
结果(准确率)：0.738
时间：28.6643168926239
'''
import  numpy as np
import time

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

class KNN(object):
    def __init__(self,train_data,train_label,K):
        '''
        构造函数
        :param train_data: 训练集的特征向量
        :param train_label: 训练集的类别向量
        :param K: 指定的K值
        '''
        self.train_data=train_data
        self.train_label=train_label
        self.input_num=self.train_data.shape[0]
        self.feature=self.train_data.shape[1]
        self.K=K

    def cal_distance(self,x1,x2):
        '''
        计算两个样本之间的距离，使用欧式距离
        :param x1: 第一个样本
        :param x2: 第二步样本
        :return: 样本之间的距离
        '''
        return np.sqrt(np.sum(np.square(x1-x2)))

    def get_K(self,x):
        dist_group=np.zeros(self.input_num)
        for i in range(self.input_num):
            x1=self.train_data[i]
            dist=self.cal_distance(x,x1)
            dist_group[i]=dist

        topK=np.argsort(dist_group)[:self.K]#升序排序

        labeldist=np.zeros(10)#10个标签，在每一个标签对应的位置上加1

        for i in range(len(topK)):
            labeldist[int(self.train_label[topK[i]])]+=1

        return np.argmax(labeldist)

    def test(self,test_data,test_label):
        '''
        在测试集上测试
        :param test_data: 测试集的特征向量
        :param test_label: 测试集的标签向量
        :return: 准确率
        '''
        error=0

        test_num=test_data.shape[0]
        for i in range(test_num):
            print(f"the current sample is {i+1},the total samples is{test_num}.")
            x=test_data[i]
            y=self.get_K(x)

            if(y!=test_label[i]):
                error+=1

        accuracy=(test_num-error)/test_num
        return accuracy

if __name__=="__main__":
    start=time.time()

    print("start load data.")
    train_data,train_label=loadData("../MnistData/mnist_train.csv")
    test_data,test_label=loadData("../MnistData/mnist_test.csv")
    print("finished load data.")

    a=KNN(train_data[:1000],train_label[:1000],30)

    print("finished training.")

    accuracy=a.test(test_data[:1000],test_label[:1000])
    print(f"the accuracy is {accuracy}.")

    end=time.time()

    print(f"the total time is {end-start}.")


