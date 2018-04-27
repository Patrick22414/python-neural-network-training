import numpy as np
from util import layer


class SimpleNet:
    def __init__(self, n_layer: int, layer_type: list, scale: list, batch_size, step_size):
        self.n = n_layer
        self.b = batch_size
        self.La = [None] * n_layer  # FIXME I don't think this is a good way to init things
        self.loss = np.zeros(batch_size)
        self.downstream_result = [np.array([])] * n_layer
        self.upstream_gradient = [np.array([])] * (n_layer+1)
        for k in range(n_layer):
            self.La[k] = layer.FCLayer(scale[k], scale[k+1], batch_size, step_size, layer_type[k])
            self.downstream_result[k] = np.zeros([scale[k+1], batch_size])
            self.upstream_gradient[k] = np.zeros([scale[k+1], batch_size])
        self.upstream_gradient[-1] = np.zeros([scale[-1], batch_size])
        print("--- Simple Net initialized with {} layer(s) ---".format(self.n))

    def train(self, data_batch, label_batch):
        # compute downstream
        self.downstream_result[0] = self.La[0].train(data_batch)
        for k in range(1, self.n):
            self.downstream_result[k] = self.La[k].train(self.downstream_result[k-1])

        # compute loss
        self.loss = - np.log(self.downstream_result[-1][label_batch, np.arange(self.b)])

        # compute back-propagation
        self.upstream_gradient[-1].fill(0)
        self.upstream_gradient[-1][label_batch, np.arange(self.b)] = -np.reciprocal(self.loss)
        for k in range(self.n).__reversed__():
            self.upstream_gradient[k] = self.La[k].backprop(self.upstream_gradient[k+1])
        # print(self.upstream_gradient)

    def predict(self, data):
        self.downstream_result[0] = self.La[0].predict(data)
        for k in range(1, self.n):
            self.downstream_result[k] = self.La[k].predict(self.downstream_result[k-1])
        return self.downstream_result[-1]
