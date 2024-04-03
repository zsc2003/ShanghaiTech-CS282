import matplotlib.pyplot as plt
import numpy as np
import tqdm

class Trainer:
    def __init__(self, data):
        self.training_data = data.training_data
        self.valid_data = data.valid_data
        self.testing_data = data.testing_data

        self.epoch = 1000
        self.eta = 0.1

        self.loss = []
        self.lr = []
        self.w_norm = []

    def get_loss(self, x, y, w):
        # logistic loss
        # X: (N, D), y: (N,), w: (D,) -> scalar
        return np.mean(np.log(1 + np.exp(-y * np.dot(x, w))))

    def get_gradient(self, x, y, w):
        # logistic gradient
        # X: (N, D), y: (N,), w: (D,) -> (D,)
        return -np.mean(y[:, np.newaxis] * x / (1 + np.exp(y * np.dot(x, w))), axis=0)
    
    def train(self):
        w = np.zeros
        
        # eta_t = eta * |grad_t|

        for _ in tqdm.tqdm(range(self.epoch)):
            gradient_t = self.get_gradient()

        pass



    def print_info(self):
        plt.figure(1)
        plt.plot(self.loss)
        plt.yscale('log')
        plt.title('Loss')
        plt.show()

        plt.figure(2)
        plt.plot(self.lr)
        plt.yscale('log')
        plt.title('Learning Rate')
        plt.show()

        plt.figure(3)
        plt.plot(self.w_norm)
        plt.title('Weight Norm')
        plt.show()



