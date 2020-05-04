'''
Author: ChrisChan
Date:  2020-4-25
二分类 — 逻辑回归
数据集：鸢尾花数据集
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from DNNtools import Loss_function

data = datasets.load_iris()
X_train = data['data'][0:100].reshape(-1,4)
y_train = data['target'][0:100].reshape(-1,1)
n_sample,n_dim = X_train.shape
learning_rate = 0.001
# colors = ['r' if l == 0 else 'b' for l in y_train[:]]
# plt.scatter(X_train[:,0],X_train[:,1],c=colors)

def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res


def cross_entropy(y_pred,y_true):
    res = np.average(- y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
    return res


def forward(x,W):
    res = sigmoid(x.dot(W))
    return res


def backpro(y_pred,y_true):
    res_w = 1/n_sample*(X_train.T.dot((y_pred-y_true)))
    # res_b = 1/n_sample*(y_pred-y_true)
    return res_w


w = np.random.randn(4,1)
# b = np.zeros(1)
for i in range(2000):
    y_pred = forward(X_train,w)
    # loss = cross_entropy(y_pred,y_train)
    loss = Loss_function.cross_entropy(y_pred,y_train)
    grad_w = backpro(y_pred,y_train)
    w = w - learning_rate * grad_w
    # b = b - learning_rate * grad_b

y_pred = (y_pred >0.5).astype(np.int)

Accuracy = (np.sum((y_pred == y_train).astype(np.int))) / len(y_train)

print(Accuracy)























