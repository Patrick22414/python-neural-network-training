import pickle


def unpickle(batch_name):
    file_name = 'F:\\Documents\\Projects_IDEA\\cifar-10\\' + batch_name
    with open(file_name, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


def test():
    batch_name = 'data_batch_1'
    batch_1 = unpickle(batch_name)
    print(batch_1.keys())
    print(batch_1[b'data'].shape)
    print(batch_1[b'labels'])


if __name__ == '__main__':
    test()
