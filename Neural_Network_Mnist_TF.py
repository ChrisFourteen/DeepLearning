'''
Author:ChrisChan
Date:2020-5-1
数据集：Mnist手写数字识别
基于 Tensorflow
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True


Mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
X_train = Mnist.train.images
y_train = Mnist.train.labels


# 定义网络参数
n_sample,n_dim = X_train.shape
n_classes = 10
Epoch_total = 200
batch_size = 55
learning_rate = 0.01

# 占位符
X_input = tf.placeholder(tf.float32,shape=([None,n_dim]))
y_input = tf.placeholder(tf.float32,shape=([None,n_classes]))

# 定义参数
weight = {
    'h1':tf.Variable(tf.random.normal((n_dim,300),mean=0,stddev=1)),
    'h2':tf.Variable(tf.random.normal((300,150),mean=0,stddev=1)),
    'h_o':tf.Variable(tf.random.normal((150,n_classes),mean=0,stddev=1))
}

bias = {
    'b1':tf.Variable(tf.zeros([300])),
    'b2':tf.Variable(tf.zeros([150])),
    'b_o':tf.Variable(tf.zeros([n_classes]))
}

# 定义网络结构
y_1 = tf.nn.relu(tf.add(tf.matmul(X_input,weight['h1']),bias['b1']))
y_2 = tf.nn.relu(tf.add(tf.matmul(y_1,weight['h2']),bias['b2']))
y_out = tf.add(tf.matmul(y_2,weight['h_o']),bias['b_o'])

y_pred = tf.nn.softmax(y_out)

# 损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y_input))

# 训练
opt = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

with tf.Session(config=config) as sess:
    def next_batch(input):
        for i in range(np.int(len(input) / batch_size)):
            yield input[i * batch_size : (i+1) * batch_size]
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    epoch_plt = []
    loss_plt = []
    acc_plt = []
    for epoch in range(Epoch_total):
        total_cost = 0
        y_p= np.array([])
        X = next_batch(X_train)
        y = next_batch(y_train)
        for i in range(np.int(len(y_train) / batch_size)):
            cost,p,_ = sess.run([loss,y_pred,opt],feed_dict={X_input:next(X),y_input:next(y)})
            total_cost +=cost
            y_p = np.append(y_p,p)
        acc = (np.sum((np.argmax(y_p.reshape(-1,n_classes),axis=1) == np.argmax(y_train,axis=1))).astype(np.int)) / n_sample
        cost = total_cost / np.int(len(y_train) / batch_size)
        acc_plt.append(acc)
        loss_plt.append(cost)
        epoch_plt.append(epoch)
        print('Epoch: %s , cost: %s , Accuracy: %s'%(epoch,cost,acc))
    stop_time = time.time()
    print('Total : %s' %(stop_time - start_time))
    plt.plot(epoch_plt,loss_plt,'r')
    plt.plot(epoch_plt,acc_plt,'g')
    plt.legend('Loss','Accuracy')
    plt.show()








