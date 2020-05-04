'''
Author:ChrisChan
Date:2020-4-27
数据集：sklearn手写数字
softmax多分类
'''
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import time

np.random.seed(10)
data = load_digits()
X_train = data['data']
y_train = data['target']
# X_train,X_test = train_test_split(X,test_size=0.2)
# y_train,y_test = train_test_split(y,test_size=0.2)
y_train = (np.arange(10) == y_train.reshape(-1,1)).astype(np.int)
# y_test = (np.arange(10) == y_test.reshape(-1,1)).astype(np.int)
n_class = 10
learning_rate = 0.005
epoch = 3000
def timer(func):
    def warper(*args,**kwargs):
        start_time = time.time()
        res = func(*args,**kwargs)
        end_time = time.time()
        print('训练时间：%s' %(end_time-start_time))
        return res
    return warper

class muti_classify:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.n_sample,self.n_dim = self.x.shape
        self.w = np.random.randn(self.n_dim,n_class)

    def forward(self):
        res = self.x.dot(self.w)
        return res

    def softmax(self):
        e = np.exp(self.forward())
        s = np.sum(e,axis=1)
        for i in range(len(s)):
            e[i] = e[i] / s[i]
        return e

    def backpro(self):
        self.y_pred = self.softmax()
        res = (1 / self.n_sample) * self.x.T.dot(self.y_pred - self.y)
        return res

    def loss(self):
        res = []
        for i in range(len(self.y)):
            a = 0
            for j in range(10):
                a += self.y[i][j] * np.log(self.y_pred[i][j])
            res.append(a)
        sum = np.sum(res)
        result = - 1 / self.n_sample * sum
        return result

    @timer
    def fit(self,learning_rate,epoch):
        loss = []
        Epoch = []
        Accuracy = []
        for i in range(epoch):
            grad = self.backpro()
            self.w = self.w - learning_rate * grad
            l = self.loss()
            Acc = np.sum((np.argmax(self.y,axis=1) == np.argmax(self.y_pred,axis=1)).astype(np.int)) / self.n_sample
            Epoch.append(i)
            loss.append(l)
            Accuracy.append(Acc)
            print('Epoch: %s , loss: %s , Acc: %s' %(i,l,Acc))
        plt.plot(Epoch,loss,'r')
        plt.plot(Epoch, Accuracy, 'g')
        return self.y_pred

obj = muti_classify(X_train,y_train)
y_pred = obj.fit(learning_rate,epoch)










