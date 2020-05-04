'''
Author:ChrisChan
Date:2020-4-27
数据集：鸢尾花数据集
Softmax主要用于处理多分类问题
'''
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from DNNtools import Loss_function
np.random.seed(10)

data = load_iris()
X_train = data['data']
y_train = data['target'].reshape(-1,1)
n_sample,n_dim = X_train.shape
learning_rate = 0.001
# 这里是one-hot常用的一个方法

y_train = (np.arange(3) == y_train).astype(np.int)
w = np.random.randn(n_dim,3)
# 可视化
# color = []
# for i in y_train[:]:
#     if i == 0:
#         color.append('r')
#     if i == 1:
#         color.append('b')
#     if i == 2:
#         color.append('g')
# plt.scatter(X_train[:,1],X_train[:,2],c=color)
# 这里的color是一个数组，这里不是靠X和y的关系判断的，仅仅是按照顺序组成一个列表

def sigmoid(x):
    res = 1 / (1+np.exp(-x))
    return res


def forward(x,weight):
    res = x.dot(weight)
    return res


def softmax(x):
    res = np.exp(x)
    sum = np.sum(res,axis=1)
    for i in range(len(sum)):
        res[i] = res[i]/sum[i]
    return res

def get_loss(y_pred,y_true):
    res = []
    for i in range(len(y_pred)):
        a = 0
        for j in range(3):
            a += y_true[i][j]*np.log(y_pred[i][j])
        res.append(a)
    res = - 1/n_sample * (np.sum(res))
    return res

def backpro(y_pred,y_true):
    res = (X_train.T.dot(y_pred-y_true)) / n_sample
    return res


epoch = []
l_plt = []
acc_plt = []
for i in range(2000):
    y_pred = forward(X_train,w)
    y_pred = softmax(y_pred)
    # loss = get_loss(y_pred,y_train)
    obj = Loss_function.cross_entropy
    loss = obj.value(obj,y_pred,y_train)
    acc = np.sum((np.argmax(y_pred, axis=1) == np.argmax(y_train, axis=1)).astype(np.int)) / n_sample
    grad = backpro(y_pred,y_train)
    w = w - learning_rate * grad
    epoch.append(i)
    l_plt.append(loss)
    acc_plt.append(acc)

print('loss: %s\nAccuracy:%s' %(loss,acc))
plt.plot(epoch,l_plt,'g')
plt.plot(epoch,acc_plt,'r')
plt.xlabel('Epoch')
plt.legend(['cost','Accuracy'])



