#coding:utf-8
#Author:codewithzichao
#E-mail:lizichao@pku.edu.cn

# mnist_train:60000
# mnist_test:10000
# acc: 0.8636
# time: 583.6889300346375


import pandas as pd
import numpy as np
import time
from collections import Counter



def loadData(fileName):
    #从文件中读取数据
    data=pd.read_csv(fileName,header=None)
    # 将数据从dataframe转化为ndarray
    data=data.values
    #数据第一行为分类结果
    y_label=data[:,0]
    x_label=data[:,1:]

    #数据二值化，返回数据
    #因为xi的取值范围为0-255，则计算p(X=xi\Y=y)的时候可能性过多，计算过于繁杂
    # 所以进行二值化
    # y_label为np.ndarray,x_label为np.ndarray

    x_label[x_label<128]=0
    x_label[x_label>=128]=1

    # mp.ndarray
    return x_label,y_label

# 计算每一列的信息熵
def calcul_H_D(column):
    '''
    :param column: 需要求信息增益的列
    :return: 信息熵
    '''
    # 计算这一列有几种取值
    types=set([i for i in column]) # set中不包含相同元素

    type_dic={} #用来计数每个Di有多少种
    HD=0
    # 初始化type_dic

    for i in types:
        type_dic[i]=0
    # HD=(Di)/D * log(Di/D)
    for i in range(len(column)):
        type_dic[column[i]]+=1
    for i in type_dic:
        HD=HD+(-1)*type_dic[i]/len(column)*np.log2(type_dic[i]/len(column))
    return HD


# 计算条件熵
# H_D_A=Di/D*H(Di
def calcul_H_D_A(column, y_label):
    '''
    :param column: 特征A所在列  需要np.array
    :param y_label: 分类结果类，D 需要np.array
    :return: 条件熵
    '''

    #计算特征A的几种取值
    types=set([i for i in column])

    # 计算出特征Ai的条件下的信息熵
    H_D_Ai={}

    type_dic = {}  # 用来计数每个Di有多少种
    for i in types:
        #初始化type_dic
        type_dic[i]=0

        # 计算特定Ai条件下的条件熵
        # y_label[column==i]得到y_label中A中特征为Ai的分类结果
        H_D_Ai[i]=calcul_H_D(y_label[column == i])

    # 用于计算出得到Di，计算Di/D
    for i in range(len(column)):
        type_dic[column[i]]+=1

    # 计算条件熵
    H_D_A=0
    for i in types:
        H_D_A+=type_dic[i]/len(column)*H_D_Ai[i]
    return H_D_A


# 找到信息增益最大的列
def findMaxFeature(X_trian,y_train):
    '''
    :param X_trian: 训练集D
    :param y_train: 训练集标签
    :return: 列
    '''

    features=X_trian.shape[1]

    H_D=0
    H_D_A=0
    max_Gain=-10000 #最大信息增益
    max_feature=-1 #最大信息增益的列

    # 样本的熵
    H_D = calcul_H_D(y_train)

    for feature in range(features): # 对列进行遍历
        # 注意是X_trian[:, feature]，别忘了：定位行
        H_D_A=calcul_H_D_A(X_trian[:, feature], y_train)

        if H_D-H_D_A>max_Gain:
            max_Gain=H_D-H_D_A
            max_feature=feature
    return max_feature,max_Gain


# 对于一列数据，找到出现最多的类，作为这一列的标记
def findCluster(column):
    # 使用counter，对每一个出现的特征计数
    ans=Counter(column)
    # 找到出现次数第一多的
    cluster=ans.most_common(1)[0][0]
    return cluster


# 对于样本根据特征进行切分
def cutData(X_train,y_train,Ag,ai):
    '''
    :param X_train: 训练样本
    :param y_train: 样本标签
    :param Ag: 需要切分特征所在的列
    :param ai: 切分特征
    :return: 切分后的训练样本，标签
    '''

    rest_train_data=[] #切分之后的训练集
    rest_train_label=[] #切分之后的标签


    for i in range(len(X_train)):
        if X_train[i][Ag]==ai:
            # a = np.array([[1, 2, 3], [1, 2, 3]])
            # b = np.array([[1, 2, 3], [4, 5, 6]])
            # a + b
            # out:array([[2, 4, 6],
            #            [5, 7, 9]])
            # 对样本进行切分，依据Ag列的ai特征
            # 切分完之后的样本没有了Ag列
            # 总行数为Ag中ai特征的行


            rest_train_data.append(list(X_train[i][0:Ag])+list(X_train[i][Ag+1:]))
            rest_train_label.append(y_train[i])
    return np.array(rest_train_data),np.array(rest_train_label)



def creTree(X_train,y_train):
    # 当信息增益小于0.3，就置T为单节点树
    epsilon=0.1

    print(f'create tree,data_length={len(X_train)}')

    # 查看总共还有多少分类
    clusters=set([i for i in y_train])

    # 若果样本中所有实例都是同一类，则T为单节点树，返回该类作为节点的标记
    if len(clusters)==1:
        # y_train中所有分类都是一样的，直接返回第一个
        return y_train[0]

    # 如果样本D中特征A为空集，则直接返回分类中最多的一类
    # X_train[0]==0 就代表没有列了
    if len(X_train[0])==0:
        return findCluster(y_train)

    # 找到最大的信息增益的列
    feature,gain=findMaxFeature(X_train,y_train)

    #若信息增益小于epsilon，则T为单节点树，返回其中最大的类作为标记
    if gain<epsilon:
        return findCluster(y_train)

    # 当信息增益大于epsilon，对样本依据特征划分子空间,递归构造子树

    # 计算这一列有几种分类
    types=set([i for i in X_train[:,feature]])

    tree_dic = {feature:{}}
    # 使用字典描述树，如tree{123:{0：7,{1:{....}}}
    # 就代表123列的0特征可以分类为7，1则继续构造子树

    for i in types:
        # 返回的是一个元组
        rest_X_train,rest_y_train=cutData(X_train, y_train, feature, i)
        tree_dic[feature][i]=creTree(rest_X_train,rest_y_train)

    return tree_dic

def predict(x_test,tree):



    while True:# 一直循环，直到在tree中找到位置

        # 得到树中的分类特征，依据分类结果
        # print(tree)

        (key, value), = tree.items()
        if type(value).__name__=='dict':
            # 如果值仍为字典，则我们需要继续遍历
            # 在对测试集继续遍历的时候，我们需要删除该分类特征（key），
            # 因为我们在构造树的时候，删除了一些特征，
            # 因此我们的到的feature也是相对的

            feature=x_test[key]
            #print(type(x_test))
            #print(x_test[key])

            # 注意x_test需要为list，才可以用del
            del x_test[key]
            # 向子树搜寻
            # 注意是value【feature】 不是tree【feature】
            tree=value[feature]
            # 子树为单节点，直接返回值
            #print(type(tree)) # numpy.int64
            #print(type(tree).__name__) # int64
            if type(tree).__name__=='int64':
                return tree
        else:
            # 若value不是字典类型
            return value

def test(X_test,y_test,tree):
    acc_num=0
    acc=0
    for i in range(len(X_test)):
        y_pred=predict(list(X_test[i]),tree)
        if y_pred==y_test[i]:
            acc_num+=1
        print(f'find {i}th data cluster:y_pred={y_pred},y={y_test[i]}')
        print('now_acc=', acc_num / (i + 1))


if __name__=="__main__":
    # 获取当前时间
    start = time.time()

    # 读取训练文件
    print("load train data")
    X_train, y_train = loadData('../MnistData/mnist_train.csv')

    # 读取测试文件
    print('load test data')
    X_test, y_test = loadData('../MnistData/mnist_test.csv')


    tree=creTree(X_train,y_train)


    test(X_test, y_test,tree)

    # 获取结束时间
    end = time.time()

    print('run time:', end - start)
