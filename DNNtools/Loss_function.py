'''
损失函数
'''
import numpy as np


class square_loss:

    def value(self,y_pred,y_true):
        num = len(y_pred)
        res = np.sum((y_pred - y_true)**2) / num
        return res

    def backpro(self,y_pred,y_true):
        res = 2 * (y_pred - y_true)
        return res


class cross_entropy:
    def __init__(self,y_pred,y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def value(self):
        res = - np.average(np.sum(self.y_true * np.log(self.y_pred),axis=1))
        return res

    def backpro(self):
        res = self.y_pred - self.y_true
        return res







