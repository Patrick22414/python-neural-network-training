import numpy as np
from util import cifar_10, crossent
import time

n_pixel = 3072 + 1
n_class = 10
step_size = 1e-4

classifier = crossent.CrossEnt(n_class=n_class, n_pixel=n_pixel, step_size=step_size)

batch = [None]*5
batch[0] = cifar_10.unpickle("data_batch_1")
batch[1] = cifar_10.unpickle("data_batch_2")
batch[2] = cifar_10.unpickle("data_batch_3")
batch[3] = cifar_10.unpickle("data_batch_4")
batch[4] = cifar_10.unpickle("data_batch_5")
n_train = 10000

start = time.time()
for usage in range(5):
    for j in range(5):
        data = np.append((batch[j][b'data'] / 256), np.ones([10000, 1]), axis=1)
        label = batch[j][b'labels']
        for k in range(n_train):
            classifier.train(data[k], label[k])
            classifier.backprop()
print("--- Training time: {0:.4f}s".format((time.time()-start)))

batch_test = cifar_10.unpickle("test_batch")
data_test = np.append((batch_test[b'data'] / 256), np.ones([10000, 1]), axis=1)
label_test = batch_test[b'labels']
n_bingo = 0
n_test = 10000

start = time.time()
for k in range(n_test):
    scores = classifier.predict(data_test[k])
    if np.argmax(scores) == label_test[k]:
        n_bingo += 1
print("--- Testing time: {0:.4f}s".format((time.time() - start)))

final_weights = classifier.w
print("--- Accuracy on test data: {}%".format(n_bingo*100/n_test))
