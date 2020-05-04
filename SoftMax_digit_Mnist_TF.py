'''
Author：ChrisChan
data:2020-4-29
数据集：Mnist手写数字
SoftMax多分类
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from DNNtools import Loss_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth = True

Mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
# Mnist数据集中的手写数字  比  sklearn中的手写数字  数据集数量更大，数据维度更大

X_train = Mnist.train.images
y_train = Mnist.train.labels



n_dim = X_train.shape[1]
n_classes = 10                                              # 指定类别数量
Epoch_total = 200
Batch_size = 55
learning_rate = 0.01


# 指定生成器
def Next_batch(list,batch_size):
    for i in range(int(len(list) / batch_size)):
        yield list[i * batch_size : (i+1) * batch_size]

class muti_classify:
    def __init__(self,X_input,y_input,learning_rate,n_classes,n_dim):
        self.X = X_input
        self.y = y_input
        self.X_input = tf.placeholder(tf.float32, shape=[None, n_dim])
        self.y_input = tf.placeholder(tf.float32, shape=[None, n_classes])
        self.eta = learning_rate
        self.W = tf.Variable(tf.random.normal((n_dim, n_classes), mean=0, stddev=1))
        self.b = tf.Variable(tf.zeros([1]))

    def forward(self):
        res = tf.add(tf.matmul(self.X_input,self.W),self.b)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=res,labels=self.y_input))
        pred = tf.nn.softmax(res)
        return cost,pred

    def fit(self):
        self.cost,self.pred = self.forward()
        opt = tf.train.GradientDescentOptimizer(self.eta).minimize(self.cost)
        Epoch_plt,loss_plt,Acc_plt = [],[],[]
        with tf.Session(config=config) as sess:
            np.random.seed(10)
            sess.run(tf.global_variables_initializer())
            for i in range(Epoch_total):
                batch_imgs = Next_batch(self.X, Batch_size)
                batch_labs = Next_batch(self.y,Batch_size)
                pred = np.array([])
                cost = 0
                for j in range(int((len(self.X) / Batch_size))):
                    _,l,p = sess.run([opt,self.cost,self.pred],feed_dict={self.X_input: next(batch_imgs),self.y_input: next(batch_labs)})
                    cost = cost + l
                    pred = np.append(pred,p).reshape(-1,10)

                acc = np.sum((np.argmax(pred,axis=1) == (np.argmax(self.y,axis=1))).astype(np.int))/ len(self.y)
                Epoch_plt.append(i)
                loss_plt.append(cost / (len(self.X) / Batch_size))
                Acc_plt.append(acc)
                print('Epoch : %s , cost : %s , Accuracy: %s'%(i,cost / (len(self.X) / Batch_size),acc))
                plt.plot(Epoch_plt,loss_plt,'r')
                plt.plot(Epoch_plt,Acc_plt,'g')
        return pred


obj = muti_classify(X_train,y_train,learning_rate,n_classes,n_dim)
y_pred = obj.fit()





















