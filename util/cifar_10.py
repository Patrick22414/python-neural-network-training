import pickle
import numpy as np


def unpickle(batch_name):
    if batch_name == "all":
        file_name = "F:/Documents/Projects_IDEA/cifar-10/data_batch_1"
        with open(file_name, 'rb') as fo:
            cifar_dict = pickle.load(fo, encoding='bytes')
        data, labels = (cifar_dict[b'data'], cifar_dict[b'labels'])

        for k in range(2, 6):
            file_name = "F:/Documents/Projects_IDEA/cifar-10/data_batch_{}".format(k)
            with open(file_name, 'rb') as fo:
                cifar_dict = pickle.load(fo, encoding='bytes')
            data = np.concatenate((data, cifar_dict[b'data']))
            labels = np.append(labels, cifar_dict[b'labels'])
        return data, labels
    else:
        file_name = "F:/Documents/Projects_IDEA/cifar-10/" + batch_name
        with open(file_name, 'rb') as fo:
            cifar_dict = pickle.load(fo, encoding='bytes')
        return cifar_dict


def test():
    name = "all"
    data, labels = unpickle(name)
    print(data.shape)
    print(labels.shape)


if __name__ == '__main__':
    test()
