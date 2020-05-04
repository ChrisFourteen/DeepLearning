'''
Author: ChrisChan
Date: 2020-4-30
数据集: Mnist手写数字识别
神经网络的重点是 反向传播
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from DNNtools import Sigmoid
from DNNtools import Relu
from DNNtools import Network
from DNNtools import Loss_function
from DNNtools import Softmax
from DNNtools import Identify
import time

Mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
X_train = Mnist.train.images
y_train = Mnist.train.labels
n_sample,input_dim = X_train.shape
n_classes = 10
learning_rate = 0.01
Epoch_total = 200
batch_size = 55

# 构建网络
layers = []
layers.append(Network.layer(input_dim,300,activation=Relu))
layers.append(Network.layer(300,100,activation=Relu))
layers.append(Network.layer(100,n_classes,activation=Identify))


# 正向传播
def fit(layers,input):
    layer_input = input
    for layer in layers:
        layer_output = layer.forward(layer_input)
        layer_input = layer_output
    return layer_input

def backpro(learning_rate,delta_in):
    layer_deltain = delta_in
    for layer in layers[::-1]:
        layer_deltaout = layer.backpro(layer_deltain,learning_rate)
        layer_deltain = layer_deltaout

def next_batch(X_input,batch_size):
    for i in range(np.int(len(X_input) / batch_size)):
        yield X_input[i * batch_size : (i+1) * batch_size]



cost_plt = []
acc_plt = []
epoch_plt = []
start_time = time.time()
for epoch in range(Epoch_total):
    cost,acc = [0,0]
    y_p = np.asarray([])
    X = next_batch(X_train,batch_size)
    y = next_batch(y_train,batch_size)
    for i in range(np.int(len(X_train) / batch_size)):
        X_input = next(X)
        y_input = next(y)
        y_pred = Softmax.softmax(fit(layers,X_input))
        loss = Loss_function.cross_entropy(y_pred,y_input).value()
        delta_in = Loss_function.cross_entropy(y_pred,y_input).backpro()
        backpro(learning_rate,delta_in)
        cost += loss
        y_p = np.append(y_p,y_pred)
    y_p = y_p.reshape(-1,n_classes)
    acc = np.sum((np.argmax(y_p, axis=1) == np.argmax(y_train, axis=1)).astype(np.int)) / n_sample
    cost = cost / np.int(len(X_train) / batch_size)
    cost_plt.append(cost)
    epoch_plt.append(epoch)
    acc_plt.append(acc_plt)
    print('Epoch : %s , cost : %s ,acc: %s'% (epoch,cost,acc))
    # delta_2 =layers[2].backpro(delta_in,learning_rate)
    # delya_3 = layers[1].backpro(delta_2,learning_rate)
    # layers[0].backpro(delya_3,learning_rate)
stop_time = time.time()
print('total_time: %s' %(stop_time-start_time))
plt.plot(epoch_plt,cost_plt)
plt.plot(epoch_plt,acc_plt)
plt.legend('cost','accuracy')
plt.show()









