'''
Author:ChrisChan
Date:2020-4-26
Tensorflow框架下的逻辑回归
数据集:鸢尾花数据集
'''
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'   #指定第一块GPU可用
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 程序最多只能占用指定gpu50%的显存
# config.gpu_options.allow_growth = True

data = load_iris()
X_train = data['data'][0:100]
Y_train = data['target'][0:100].reshape(-1,1)

n_sample,n_dim = X_train.shape
learning_rate = 0.01
X = tf.placeholder(tf.float32, shape=[None, n_dim])
y = tf.placeholder(tf.float32, shape=[n_sample, 1])
W = tf.Variable(tf.random.uniform([n_dim,1]))
b = tf.Variable(tf.zeros([1]))

z = tf.matmul(X,W) + b
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=z))
y_pred = tf.compat.v1.sigmoid(z)
train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

Epoch = 2000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    Epoch_time = []
    loss_plt = []
    for i in range(Epoch):
        y_p,l,_c = sess.run([y_pred,loss,train_opt],feed_dict={X:X_train,y:Y_train})
        print('Loss :{}'.format(l))
        Epoch_time.append(i)
        loss_plt.append(l)

    y_p = (y_p > 0.5).astype(np.int)
    Accuracy = 1 / n_sample * (np.sum((y_p == Y_train).astype(np.int)))
    print('Accuracy: {}'.format(Accuracy))
    plt.plot(Epoch_time, loss_plt, c='r')
    plt.xlabel('epoch')
    plt.ylabel('loss')



