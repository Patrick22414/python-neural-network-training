import numpy as np


class FCLayer:
    def __init__(self, n_inputs: int, n_outputs: int, step_size: float, activation: str):
        self.i = n_inputs  # number of inputs
        self.o = n_outputs  # number of outputs
        self.act_fun = activation  # type of activation function
        self.step = step_size  # step size

        self.w = np.zeros([self.o, self.i])  # weights matrix
        self.x = np.zeros(self.i)  # input data
        self.s = np.zeros(self.o)  # s = w.dot(x)
        self.y = np.zeros(self.o)  # output

        self.s_exp = self.s

    def train(self, data: np.ndarray):
        self.x = data

        if self.act_fun == "softmax":
            return self.softmax()
        else:
            raise NotImplementedError

    def backprop(self, label, previous_grad=None):
        if previous_grad is None:
            previous_grad = np.ones(self.o)

        if self.act_fun == "softmax":
            return self.softmax_bp(label, previous_grad)
        else:
            raise NotImplementedError

    def predict(self, data):
        tmp = np.exp(np.matmul(self.w, data))
        res = tmp / np.sum(tmp)
        return res

    def softmax(self):
        """softmax forward compute"""
        self.s = np.matmul(self.w, self.x)
        self.s_exp = np.exp(self.s)
        self.y = self.s_exp / np.sum(self.s_exp)
        return self.y

    def softmax_bp(self, label, previous_grad):
        """softmax back-propagation"""
        # TODO backprop should work for all y instead of y[label]
        dy = - previous_grad / self.y[label]

        ds = dy * ((- self.s_exp[label] * self.s_exp) / (np.sum(self.s_exp)))
        ds[label] = dy * ((np.sum(self.s_exp) - self.s_exp[label]) / (np.sum(self.s_exp)))

        dw = np.outer(ds, self.x)
        dx = np.matmul(self.w.T, self.s)

        self.w -= self.step * dw
        return dx
