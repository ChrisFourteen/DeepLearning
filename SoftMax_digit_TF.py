'''
Author:ChrisChan
date:2020-4-28
数据集: Sklearn手写数字
Tensorflow框架下的SoftMax多分类
'''
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth = True

data = load_digits()
# data = input_data.read_data_sets('MNIST_data/', one_hot=True)
X_train = data['data']
y_train = data['target'].reshape(-1,1)
y_train = (np.arange(10) == y_train).astype(np.int)

# X_train,X_test = train_test_split(X,test_size=0.2)
# y_train,y_test = train_test_split(y,test_size=0.2)
n_dim = 64
n_classes = 10
learning_rate = 0.005
Epoch = 5000

# 定义变量
X_input = tf.placeholder(tf.float32,shape=[None,n_dim])
Y_input = tf.placeholder(tf.float32,shape=[None,n_classes])

# 定义参数
W = tf.Variable(tf.random.normal([n_dim,n_classes]))
b = tf.Variable(tf.zeros([10]))

class muti_classify:
    def __init__(self,X,y,n_classes,n_dim,learning_rate):
        self.X = X
        self.y = y
        self.dim = n_dim
        self.eta = learning_rate
        self.W = tf.compat.v1.Variable(tf.random.uniform([self.dim,n_classes]))

    def forward(self,X):
        res = tf.nn.softmax(tf.matmul(X,self.W))
        return res

    def loss(self,X):
        y_pred = tf.matmul(X,self.W)
        res = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=y_pred))
        return res

    def fit(self,epoch=3000):
        X = tf.placeholder(tf.float32,shape=[None,self.dim])
        y = tf.placeholder(tf.float32, shape=[None, n_classes])
        l = self.loss(X)
        y_p = self.forward(X)
        opt = tf.train.GradientDescentOptimizer(self.eta).minimize(l)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                y_pred,loss,_c = sess.run([y_p,l,opt],feed_dict={X:self.X,y:self.y})
                print('epoch: %s , loss: %s'%(i,loss))
            Accuracy = np.sum((np.argmax(y_pred,axis=1) == np.argmax(self.y,axis=1)).astype(np.int)) / 1797
            print('Accuracy: ',Accuracy)
        return y_pred

obj = muti_classify(X_train,y_train,n_classes,n_dim,learning_rate)
y_pred = obj.fit()






