#coding:utf-8
#Author:codewithzichao
#E-mail:lizichao@pku.edu.cn

'''

数据集：Mnist
准确率：1.0
时间：53.53627586364746

'''

import pandas as pd
print(pd.__version__)
import numpy as np
from sklearn.model_selection import train_test_split
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os

path=os.getcwd()

start=time.time()

train=pd.read_csv(path+"/MnistData/mnist_train.csv",names=list(i for i in range(784)))
test=pd.read_csv(path+"/MnistData/mnist_test.csv",names=list(i for i in range(784)))

train_data=train.iloc[:,1:]
train_label=train.iloc[:,0]
print(train_data.shape)
print(train_label.shape)
test_data=test.iloc[:,1:]
test_label=test.iloc[:,0]


params={
'booster':'gbtree',
'objective': 'multi:softmax', #多分类的问题
'num_class':10, # 类别数，与 multisoftmax 并用
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':12, # 构建树的深度，越大越容易过拟合
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':3,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.007, # 如同学习率
'seed':1000,
'nthread':7,# cpu 线程数
#'eval_metric': 'auc'
}

num_rounds = 5000 # 迭代次数


X_train,X_validation,Y_train,Y_validation= train_test_split(train_data,train_label, test_size = 0.3,random_state=1)
#random_state is of big influence for val-auc


xgb_val = xgb.DMatrix(X_validation,label=Y_validation)
xgb_train = xgb.DMatrix(X_train, label=Y_train)
xgb_test = xgb.DMatrix(test_data)


watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]#允许查看train set与dev set的误差表现
# training model
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(params,xgb_train, num_rounds, watchlist,early_stopping_rounds=30)

model.save_model(path+"/xgboost/xgb.model") # 用于存储训练出的模型
print ("best best_ntree_limit",model.best_ntree_limit)

print ("跑到这里了model.predict")
preds = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)

accuracy_test=accuracy_score(preds,test_label)
print(f"the accuracy is {accuracy_test}.")

end=time.time()
#输出运行时长

print (f"xgboost success! cost time:{end-start}(s).")
