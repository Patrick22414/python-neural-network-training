import numpy as np
from util import cifar_10, nets
import time

n_layer = 2
scale = [3072, 128, 10]
layer_type = ["ReLU", "Softmax"]
batch_size = 10000
step_size = 1

net = nets.SimpleNet(n_layer, layer_type, scale, batch_size, step_size)

batch = [None] * 5
batch[0] = cifar_10.unpickle("data_batch_1")
batch[1] = cifar_10.unpickle("data_batch_2")
batch[2] = cifar_10.unpickle("data_batch_3")
batch[3] = cifar_10.unpickle("data_batch_4")
batch[4] = cifar_10.unpickle("data_batch_5")

for usage in range(10):
    for j in range(5):
        data = batch[j][b'data'] / 256
        label = batch[j][b'labels']
        net.train(data.T, label)

test_batch = cifar_10.unpickle("test_batch")
test_data = test_batch[b'data'].T
test_label = test_batch[b'labels']

result = np.argmax(net.predict(test_data), axis=0)
n_bingo = np.sum(result == test_label)
print("--- Accuracy on test data: {:.2f} %".format(n_bingo/100))
