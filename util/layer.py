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
        self.dy = np.zeros(self.o)
        print("--- FC Layer initialized as {} layer ---", self.act_fun)

    def train(self, data: np.ndarray):
        self.x = data

        if self.act_fun == "Softmax" or self.act_fun == "Softmax loss":
            return self.softmax()
        elif self.act_fun == "ReLU":
            return self.relu()
        else:
            raise NotImplementedError

    def backprop(self, label=None, upstream_grad=None):
        if upstream_grad is not None:
            self.dy = upstream_grad

        if self.act_fun == "Softmax":
            assert (label is None) and (upstream_grad is not None)
            return self.softmax_bp()
        elif self.act_fun == "Softmax loss":
            assert (label is not None) and (upstream_grad is None)
            return self.softmax_loss_bp(label)
        elif self.act_fun == "ReLU":
            assert (label is None) and (upstream_grad is not None)
            return self.relu_bp()

    def predict(self, data):
        tmp = np.exp(np.matmul(self.w, data))
        res = tmp / np.sum(tmp)
        return res

    def softmax(self):
        """softmax forward compute"""
        self.s = np.matmul(self.w, self.x)
        s_exp = np.exp(self.s - np.max(self.s))
        self.y = s_exp / np.sum(s_exp)
        return self.y

    def softmax_loss(self, label):
        return self.softmax()[label]

    def softmax_bp(self):
        """softmax back-propagation"""
        dy_ds = np.zeros([self.o, self.o])
        for k in range(self.o):
            dy_ds[k] = - self.y[k] * self.y
            dy_ds[k, k] += self.y[k]
        ds = np.matmul(dy_ds, self.dy)

        dw = np.outer(ds, self.x)
        dx = np.matmul(self.w.T, ds)

        self.w -= self.step * dw
        return dx

    def softmax_loss_bp(self, label):
        """softmax loss back-propagation"""
        dy = -1 / self.y[label]

        ds = - self.y[label] * self.y
        ds[label] += self.y[label]
        ds *= dy

        dw = np.outer(ds, self.x)
        dx = np.matmul(self.w.T, ds)

        self.w -= self.step * dw
        return dx

    def relu(self):
        self.s = np.matmul(self.w, self.x)
        self.y = self.s
        self.y[self.y < 0] = 0
        return self.y

    def relu_bp(self):
        ds = np.ones_like(self.s)
        ds[self.s < 0] = 0
        ds *= self.dy
        dw = np.outer(ds, self.x)
        dx = np.matmul(self.w.T, ds)

        self.w -= self.step * dw
        return dx
