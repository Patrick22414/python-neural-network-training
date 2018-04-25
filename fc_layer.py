import numpy as np


class FCLayer:
    def __init__(self, i_neuron: int, o_neuron: int, activation: str):
        self.i = i_neuron  # number of inputs
        self.o = o_neuron  # number of outputs
        self.act_fun = activation  # type of activation function

        self.w = np.zeros([self.o, self.i])  # weights matrix
        self.x = np.zeros(self.i)  # input data
        self.s = np.zeros(self.o)  # s = w.dot(x)
        self.y = np.zeros(self.o)  # output

    def train(self, data, label):
        pass

    def predict(self, data):
        pass

    def softmax(self):
        """softmax forward compute"""
        pass

    def softmax_bp(self):
        """softmax back-propagation"""
        pass
