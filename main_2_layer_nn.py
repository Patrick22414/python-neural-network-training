import numpy as np
from util import cifar_10, layer
import time

n_pixel = 3072
n_mid = 100
n_class = 10
step_size = 1e-3

layer_1 = layer.FCLayer(n_pixel, n_mid, step_size, "Softmax")
layer_2 = layer.FCLayer(n_mid, n_class, step_size, "Softmax loss")

batch = [None] * 5
batch[0] = cifar_10.unpickle("data_batch_1")
batch[1] = cifar_10.unpickle("data_batch_2")
batch[2] = cifar_10.unpickle("data_batch_3")
batch[3] = cifar_10.unpickle("data_batch_4")
batch[4] = cifar_10.unpickle("data_batch_5")
n_train = 10000

# TODO rewrite CIFAR-10 test