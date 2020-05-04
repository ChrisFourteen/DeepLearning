'''
网络层定义
'''
from DNNtools import Relu,Sigmoid
import numpy as np

class layer:
    def  __init__(self,n_input,n_output,activation=Relu):
        self.r = np.sqrt(6 / (n_input + n_output))
        self.W = np.random.uniform(-self.r,self.r,(n_input,n_output))
        self.b = np.zeros(n_output)
        self.activation = activation

    def forward(self,input):
        self.input = input
        self.z = self.input.dot(self.W) + self.b
        self.output = self.activation.forward(self.z)
        return self.output

    def backpro(self,delta_in,learning_rate):
        d = delta_in * self.activation.deri(self.z)
        self.delta_out = d.dot(self.W.T)
        self.grad_w = self.input.T.dot(d)
        self.grad_b = d
        self.w = self.W - learning_rate * self.grad_w
        self.b = self.b - learning_rate * self.grad_b
        return self.delta_out




