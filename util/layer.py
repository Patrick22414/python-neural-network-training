import numpy as np


class FCLayer:
    def __init__(self, n_inputs: int, n_outputs: int, batch_size: int, step_size: float, activation: str):
        self.i = n_inputs           # number of inputs
        self.o = n_outputs          # number of outputs
        self.n = batch_size         # batch size TODO rewrite method in consideration of batches
        self.act_fun = activation   # type of activation function
        self.step = step_size       # step size

        self.w = np.zeros([self.o, self.i])     # weights matrix
        self.x = np.zeros([self.i, self.n])     # input data
        self.s = np.zeros([self.o, self.n])     # s = w.dot(x)
        self.y = np.zeros([self.o, self.n])     # output
        print("--- FC Layer initialized as {} layer ---".format(self.act_fun))

    def train(self, data: np.ndarray):
        self.x = data

        if self.act_fun == "Softmax":
            return self.softmax()
        elif self.act_fun == "ReLU":
            return self.relu()
        else:
            raise NotImplementedError

    def backprop(self, label=None, upstream_grad=None):
        pass

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

    def softmax_bp(self):
        """softmax back-propagation"""
        dy_ds = np.zeros([self.o, self.o])
        for k in range(self.o):
            dy_ds[k] = - self.y[k] * self.y
            dy_ds[k, k] += self.y[k]
        ds = np.matmul(dy_ds, self.dy)  # FIXME should upstream gradients be stored in a layer?

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
