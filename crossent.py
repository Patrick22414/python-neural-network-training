# my implementation of cross-entropy loss
import numpy as np


class CrossEnt:
    def __init__(self, n_class, n_pixel, step_size):
        self.step = step_size
        # self.w = np.random.rand(n_class, n_pixel)   # weight matrix ini random
        self.w = np.zeros((n_class, n_pixel))   # weight matrix ini 0
        self.x = np.zeros(n_pixel)              # data vector
        self.y = 0                              # correct label
        self.s = np.zeros(n_class)              # scores vector
        self.sn = np.zeros(n_class)             # normalized scores
        self.se = np.zeros(n_class)             # exp normalized scores
        self.pr = 0.0                           # probability of correct guess
        self.L = 0.0                            # loss
        print("--- CrossEnt initialized ---")

    def predict(self, data):
        return self.w.dot(data)

    def train(self, data, correct):
        self.x = data
        self.y = correct
        self.s = self.w.dot(data)
        self.sn = self.s - np.max(self.s)
        self.se = np.exp(self.sn)
        self.pr = self.se[self.y] / np.sum(self.se)
        self.L = -np.log(self.pr)

    def backprop(self):
        dpr = -1 / self.pr

        dpr_dse = -(self.se[self.y] * self.se) / (np.sum(self.se)**2)
        dpr_dse[self.y] = (np.sum(self.se) - self.se[self.y]) / (np.sum(self.se)**2)
        dse = dpr_dse * dpr

        dse_dsn = np.exp(self.sn)
        dsn = dse * dse_dsn

        ds = dsn
        ds[np.argmax(self.s)] = 0

        dw = np.outer(ds, self.x)
        self.w -= self.step * dw

