'''
Relu 激活函数 && Relu 导数
'''

import numpy as np
import matplotlib.pyplot as plt

# Relu函数
def forward(x):
    res = np.maximum(0,x)
    return res

# Relu函数的导数
def deri(x):
    res = (x>0).astype(np.int)
    return res

if __name__ == '__main__':
    x = np.arange(-10.,10.,.1)
    y_1 = [forward(i) for i in x]
    y_2 = [deri(i) for i in x]
    plt.plot(x,y_1)
    plt.plot(x,y_2)
    plt.legend(['Relu','Relu_deri'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

