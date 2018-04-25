import numpy as np


def softmax(weights, data, correct):
    """softmax, or cross-entropy loss function"""
    scores = weights.dot(data)
    scores_exp = np.exp(scores - np.max(scores))
    likelihood = scores_exp[correct] / scores_exp.sum()
    loss = -np.log(likelihood)
    return loss


def svm(weights, data, correct):
    """svm, or hinge loss function"""
    scores = weights.dot(data)
    delta = 1.0
    margins = np.maximum(0, scores - scores[correct] + delta)
    margins[correct] = 0
    loss = margins.sum()
    return loss
