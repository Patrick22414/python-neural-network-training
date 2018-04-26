import numpy as np
from util import layer


class SimpleNet:  # TODO a new class represents the whole NN with all layers
    def __init__(self, n_layer: int, layer_type: list, scale: list, batch_size, step_size):
        self.la = [None] * n_layer
        for k in range(n_layer):
            self.la[k] = layer.FCLayer(scale[k], scale[k+1], batch_size, step_size, layer_type[k])

        self.Loss = np.zeros(batch_size)

    def train(self, data_batch, label):  # TODO a single method to train the whole net
        pass

    def predict(self, data):
        pass