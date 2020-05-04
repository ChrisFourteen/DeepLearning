'''
Author：ChrisChan
date:2020-4-25
从最简单的线性回归写起
'''
import numpy as np
import matplotlib.pyplot as plt
from DNNtools.Loss_function import square_loss


def get_y(x):
    a = np.random.uniform(-1, -1.5)                   # 生成-1到-1.5之间的符合均匀分布的随机数 参数形式（start,end,shape）
    b = np.random.uniform(-0.5, 0.5)
    res = a * x + b
    return res


x_train = np.arange(0, 10., .1)                       # 生成从0到10，间隔为0.1的所有数
y_train = list(map(get_y, x_train))                   # 这里用了 python自带的map函数
plt.scatter(x_train, y_train)                         # plt画点图
# numpy.random.randn()函数  生成标准高斯分布，参数为每个维度的形状


w = 1                                                 # 初始化权值矩阵
b = 0                                                 # 初始化偏置
learn_rate = 0.001                                    # 学习率

'''前向传播函数'''
def forward(x_list):
    res = []
    for i in x_list:
        res.append(i * w + b)
    return res

'''反向传播函数'''
def backpro(y_list):
    grad_w = np.average(2 * x_train * (y_list - y_train))
    grad_b = np.average(2 * (y_list - y_train))
    return grad_w,grad_b


plt.ion()                                              # 显示模式转换为交互模式
for i in range(100):                                   # 迭代次数100
    y_perd = forward(x_train)                          # 前向传播
    x_train = np.array(x_train)                        # 将列表转换为np的array类型
    y_perd = np.array(y_perd)
    y_train = np.array(y_train)
    loss = square_loss(y_perd,y_train)
    # loss = np.average((y_perd - y_train) ** 2)         # 计算损失，这里的损失是平方损失
    print('迭代次数: {}, 损失：{}'.format(i, loss))
    grad_w,grad_b = backpro(y_perd)                    # 计算梯度
    w = w - learn_rate * grad_w                        # 更新权重
    b = w - learn_rate * grad_b                        # 更新偏置
    plt.plot(x_train, x_train * w + b, 'r')            # 画线
    plt.draw()
    plt.pause(0.5)
plt.ioff()
plt.show()
