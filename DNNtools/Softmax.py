'''
Softmax分类
'''

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e = np.exp(x)
    s = np.sum(e,axis=1)
    for i in range(len(s)):
        e[i] = e[i] / s[i]
    return e


if __name__ == '__main__':
    list = np.random.randn(1,10)
    result = np.argmax(softmax(list),axis=1)
    print(result)

