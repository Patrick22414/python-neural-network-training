# my implementation of cross-entropy loss
import numpy as np


class CrossEnt:
    def __init__(self, n_class, n_pixel, step_size):
        self.step = step_size
        # self.w = np.random.rand(n_class, n_pixel)   # weight matrix ini as random
        self.w = np.zeros((n_class, n_pixel))  # weight matrix ini as 0
        self.x = np.zeros(n_pixel)  # data vector
        self.label = 0  # correct label
        self.s = np.zeros(n_class)  # scores vector
        self.pr = np.zeros(n_class)  # probability of correct guess
        self.L = 0.0  # loss
        print("--- CrossEnt initialized ---")

    def predict(self, data):
        return self.w.dot(data)

    def train(self, data, correct):
        self.x = data
        self.label = correct
        self.s = self.w.dot(data)
        s_exp = np.exp(self.s - np.max(self.s))
        self.pr = s_exp / np.sum(s_exp)
        self.L = -np.log(self.pr)

    def backprop(self):
        dpr = -1 / self.pr[self.label]

        ds = - self.pr[self.label] * self.pr
        ds[self.label] += self.pr[self.label]
        ds[np.argmax(self.s)] = 0
        ds *= dpr

        dw = np.outer(ds, self.x)
        self.w -= self.step * dw
        print(dw)
