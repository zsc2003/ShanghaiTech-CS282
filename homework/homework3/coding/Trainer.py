import matplotlib.pyplot as plt
import numpy as np
import tqdm


class Trainer:
    def __init__(self, data):
        self.training_data = data.training_data
        self.valid_data = data.valid_data
        self.testing_data = data.testing_data

        self.learning_rate = 0.0001
        self.epoch = 1000

        self.eta = 0.1

    def get_loss(self, x, y, w):
        # logistic loss
        # X: (N, D), y: (N,), w: (D,) -> scalar
        return np.mean(np.log(1 + np.exp(-y * np.dot(x, w))))

    def get_gradient(self, x, y, w):
        # logistic gradient
        # X: (N, D), y: (N,), w: (D,) -> (D,)
        return -np.mean(y[:, np.newaxis] * x / (1 + np.exp(y * np.dot(x, w))), axis=0)
    
    def train(self):
        # eta_t = eta * |grad_t| 
        pass