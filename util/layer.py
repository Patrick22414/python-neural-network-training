import numpy as np


class FCLayer:
    def __init__(self, n_inputs: int, n_outputs: int, batch_size: int, step_size: float, activation: str):
        self.i = n_inputs           # number of inputs
        self.o = n_outputs          # number of outputs
        self.n = batch_size         # batch size
        self.act_fun = activation   # type of activation function
        self.step = step_size       # step size

        self.w = np.zeros([self.o, self.i])     # weights matrix
        self.x = np.zeros([self.i, self.n])     # input data
        self.s = np.zeros([self.o, self.n])     # s = w.dot(x)
        self.y = np.zeros([self.o, self.n])     # output

        self.dx = np.zeros([self.i, self.n])
        self.dy = np.zeros([self.o, self.n])
        print("--- FC Layer initialized as {} layer ---".format(self.act_fun))

    def train(self, data: np.ndarray):
        self.x = data
        if self.act_fun == "Softmax":
            self.softmax()
        elif self.act_fun == "ReLU":
            self.relu()
        else:
            raise NotImplementedError
        return self.y

    def backprop(self, upstream_gradient):
        self.dy = upstream_gradient
        if self.act_fun == "Softmax":
            self.softmax_bp()
        elif self.act_fun == "ReLU":
            self.relu_bp()
        else:
            raise NotImplementedError
        return self.dx

    def predict(self, data: np.ndarray):
        if self.act_fun == "Softmax":
            return self.softmax(data=data)
        elif self.act_fun == "ReLU":
            return self.relu(data=data)

    def softmax(self, data=None):
        """softmax forward compute"""
        if data is None:
            self.s = np.matmul(self.w, self.x)
            s_exp = np.exp(self.s - np.max(self.s, axis=0))
            self.y = s_exp / np.sum(s_exp)
        else:
            s = np.matmul(self.w, data)
            s_exp = np.exp(s - np.max(s, axis=0))
            return s_exp / np.sum(s_exp)

    def softmax_bp(self):
        """softmax back-propagation"""
        ds = np.zeros_like(self.s)
        for k in range(self.n):
            dyds = - np.outer(self.y[:, k], self.y[:, k])
            dyds[np.diag_indices_from(dyds)] += self.y[:, k]
            ds[:, k] = np.matmul(dyds, self.dy[:, k])
        ds[np.argmax(self.s, axis=0), :] = 0

        dw = np.matmul(ds, self.x.T)
        self.dx = np.matmul(self.w.T, ds)
        self.w -= self.step * dw
        print(self.w)

    def relu(self, data=None):
        """rectified linear unit forward compute"""
        if data is None:
            self.s = np.matmul(self.w, self.x)
            self.y = self.s
            self.y[self.y < 0] = 0
        else:
            s = np.matmul(self.w, data)
            s[s < 0] = 0
            return s

    def relu_bp(self):
        """rectified linear unit back=propagation"""
        ds = np.ones_like(self.s)
        ds[self.s < 0] = 0
        ds = ds * self.dy

        dw = np.matmul(ds, self.x.T)
        self.dx = np.matmul(self.w.T, ds)
        self.w -= self.step * dw
