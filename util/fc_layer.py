import numpy as np


class Softmax:
    def __init__(self, n_input, n_output, batch_size, step_size, max_epoch, weights_init="Xavier"):
        # hyper-parameters
        self.I = n_input
        self.O = n_output
        self.B = batch_size
        self.Step = step_size
        self.MaxEpoch = max_epoch

        # learning parameters
        self.w = np.array([])

        # variables
        self.x = self.s = self.y = np.array([])

        # weights initialization
        if weights_init == "Xavier":
            self.w = np.random.randn(n_input, n_output) / np.sqrt(n_input)
        elif weights_init == "Zeros":
            self.w = np.zeros([n_input, n_output])
        elif weights_init == "Uniform":
            self.w = 0.1 * np.random.rand(n_input, n_output)
        else:
            raise NotImplementedError

    def predict(self, data: np.ndarray) -> np.ndarray:
        return data.dot(self.w)

    def train(self, data: np.ndarray, labels: np.ndarray, babysitter=False):
        assert data.shape == (self.B, self.I)
        assert labels.shape == (self.B,)
        self.x = data

        epoch = 0
        while epoch < self.MaxEpoch:
            epoch += 1
            self.s = self.x @ self.w  # matrix scale: B-by-I @ I-by-O -> B-by-O
            exp_s = np.exp(self.s)
            self.y = exp_s / np.sum(exp_s, axis=1)[:, np.newaxis]
            answer = self.y[np.arange(self.B), labels]

            # back-propagation
            dldy = - np.reciprocal(answer) / self.B

            dlds = - exp_s * answer[:, np.newaxis]
            dlds[np.arange(self.B), labels] += answer
            dlds *= dldy[:, np.newaxis]

            dldw = self.x.T @ dlds  # matrix scale: I-by-B @ B-by-O -> I-by-O
            self.w -= self.Step * dldw

            if babysitter:
                if epoch == self.MaxEpoch:  # comment this line to view all training process
                    loss = np.mean(- np.log(answer))
                    predicted_labels = np.argmax(self.s, axis=1)
                    validation = np.sum(predicted_labels == labels) / self.B
                    print("epoch: {:3d}/{:3d},  loss: {:.4f},  validation: {:.2f}".format(epoch,
                                                                                          self.MaxEpoch,
                                                                                          loss,
                                                                                          validation))
