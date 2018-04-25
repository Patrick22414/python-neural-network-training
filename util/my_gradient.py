"""These are really bad codes written when I first started learning Python and Neural Nets, please ignore them"""
import numpy as np
from util import my_loss


def gradient_check(loss_f, w, x, y):
    """computing the gradient numerically with finite differences"""
    grad = np.zeros(w.shape)
    delta = 0.000001

    it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        index = it.multi_index
        old_value = w[index]

        w[index] = old_value + delta
        s_inc = loss_f(w, x, y)
        w[index] = old_value - delta
        s_dec = loss_f(w, x, y)
        w[index] = old_value

        grad[index] = (s_inc - s_dec) / delta / 2
        it.iternext()

    return grad


def gradient_svm(weights, data, correct):
    grad = np.zeros(weights.shape)
    scores = weights.dot(data)
    margins = scores - scores[correct] + 1

    for index in range(weights.shape[0]):
        if index == correct:
            grad[index] = -(np.greater(margins, 0).sum() - 1) * data
        else:
            grad[index] = data if margins[index] > 0 else 0

    return grad


def test():
    weights = np.random.rand(3, 12)
    data = np.random.rand(12)
    correct = 0
    step_size = 0.1

    loss = my_loss.svm(weights, data, correct)
    grad_a = gradient_svm(weights, data, correct)
    grad_n = gradient_check(my_loss.svm, weights, data, correct)
    print('Loss:', loss)
    print('\nGradient-analytical:\n', grad_a)
    print('\nGradient-numerical:\n', grad_n)
    print('\nDifference:', grad_n-grad_a)

    weights -= step_size*grad_n
    loss_updated = my_loss.svm(weights, data, correct)
    print('\nLoss updated:', loss_updated)


if __name__ == '__main__':
    test()
