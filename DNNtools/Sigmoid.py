'''
sigmoid 激活函数  && sigmoid 导数
'''
import numpy as np
import matplotlib.pyplot as plt

# sigmoid激活函数
def forward(x):
    res = 1 / (1 + np.exp(-x))
    return res
# sigmoid的导数
def deri(x):
    res = np.exp(-x) / ((1 + np.exp(-x))**2)
    return res

# sigmoid函数  及其  导数  的图像
if __name__ == '__main__':
    x = np.arange(-10.,10.,0.1)
    y_1 = [forward(i) for i in x]
    y_2 = [deri(i) for i in x]
    plt.plot(x,y_1)
    plt.plot(x,y_2)
    plt.legend(['Sigmoid','sigmoig_deri'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

