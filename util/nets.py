import numpy as np


class TwoLayerNN:
    """Simple 2-layer neural net consisting of a ReLU and a Softmax FC layer"""

    def __init__(self, scale: tuple, batch_size, step_size, max_epoch, weights_init="Xavier"):
        assert len(scale) == 3

        # hyper-parameters
        self.Scale = scale
        self.B = batch_size
        self.Step = step_size
        self.MaxEpoch = max_epoch

        # learning parameters
        self.w1 = self.w2 = np.array([])

        # variables
        self.x1 = self.s1 = self.x2 = self.s2 = self.y = np.array([])

        # weights initialization
        if weights_init == "Xavier":
            self.w1 = np.random.randn(scale[0], scale[1]) / np.sqrt(scale[0])
            self.w2 = np.random.randn(scale[1], scale[2]) / np.sqrt(scale[1])
        elif weights_init == "Zeros":
            self.w1 = np.zeros([scale[0], scale[1]])
            self.w2 = np.zeros([scale[1], scale[2]])
        elif weights_init == "Uniform":
            self.w1 = 0.1 * np.random.rand(scale[0], scale[1])
            self.w2 = 0.1 * np.random.rand(scale[1], scale[2])
        else:
            raise NotImplementedError

    def predict(self, data: np.ndarray) -> np.ndarray:
        assert data.shape[-1] == self.Scale[0]
        return np.maximum(0, data @ self.w1) @ self.w2

    def train(self, data: np.ndarray, labels: np.ndarray, babysitter=False):
        assert data.shape == (self.B, self.Scale[0])
        assert labels.shape == (self.B,)
        self.x1 = data

        epoch = 0
        m1 = m2 = 0.0  # momentum
        while epoch < self.MaxEpoch:
            epoch += 1
            self.s1 = self.x1 @ self.w1  # matrix scale: B-by-Sc[0] @ Sc[0]-by-Sc[1] -> B-by-Sc[1]
            self.x2 = np.maximum(0, self.s1)
            # self.x2 = self.s1
            # self.x2[self.s1 < 0] *= 0.1
            self.s2 = self.x2 @ self.w2  # matrix scale: B-by-Sc[1] @ Sc[1]-by-Sc[2] -> B-by-Sc[2]
            exp_s2 = np.exp(self.s2)
            self.y = exp_s2 / np.sum(exp_s2, axis=1)[:, np.newaxis]
            answer = self.y[np.arange(self.B), labels]  # matrix scale B-by-Sc[2] -> B

            # back-propagation
            dldy = - np.reciprocal(answer) / self.B

            dlds2 = - exp_s2 * answer[:, np.newaxis]
            dlds2[np.arange(self.B), labels] += answer
            dlds2 *= dldy[:, np.newaxis]

            dldw2 = self.x2.T @ dlds2  # matrix scale: Sc[1]-by-B @ B-by-Sc[2] -> Sc[1]-by-Sc[2]
            dldx2 = dlds2 @ self.w2.T  # matrix scale: B-by-Sc[2] @ Sc[2]-by-Sc[1] -> B-by-Sc[1]

            dlds1 = dldx2
            dlds1[self.s1 < 0] = 0
            # dlds1[self.s1 < 0] *= 0.1

            dldw1 = self.x1.T @ dlds1

            m1 = 0.9 * m1 + dldw1
            m2 = 0.9 * m2 + dldw2
            self.w1 -= self.Step * m1
            self.w2 -= self.Step * m2

            if babysitter:
                if epoch == self.MaxEpoch:  # comment this line to view all training process
                    self.loss = np.mean(- np.log(answer))
                    predicted_labels = np.argmax(self.s2, axis=1)
                    validation = np.sum(predicted_labels == labels) / self.B
                    print("epoch: {:2d}/{:2d}".format(epoch, self.MaxEpoch), end='\t')
                    print("loss: {:4.2f}".format(self.loss), end='\t')
                    print("validation: {:4.2f}".format(validation), end='\t')
                    print("w1_step: {:8.2f}".format(np.square(np.sum(dldw1))), end='\t')
                    print("w2_step: {:8.2f}".format(np.square(np.sum(dldw2))))
