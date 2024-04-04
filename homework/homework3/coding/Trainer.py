import matplotlib.pyplot as plt
import numpy as np
import tqdm

class Trainer:
    from Dataloader import DataLoader
    def __init__(self, data: DataLoader):
        self.training_data = data.training_data
        self.training_labels = data.training_label

        self.testing_data = data.testing_data
        self.testing_labels = data.testing_label

        self.epoch = 1000000
        self.eta = 0.1

        self.loss = []
        self.lr = []
        self.w_norm = []

        self.dimension = self.training_data.shape[1]

    def get_loss(self, x, y, w):
        # logistic loss
        # X: (N, D), y: (N, 1), w: (D, 1) -> scalar
        return np.mean(np.log(1 + np.exp(-y * np.dot(x, w))))

    def get_gradient(self, x, y, w):
        # logistic gradient
        # X: (N, D), y: (N, 1), w: (D, 1) -> (D, 1)
        return -np.mean(y * x / (1 + np.exp(y * np.dot(x, w))), axis=0).reshape(-1, 1)
    
    def train(self):
        w = np.zeros(self.dimension).reshape(-1, 1)
        for _ in tqdm.tqdm(range(self.epoch)):
            gradient_t = self.get_gradient(self.training_data, self.training_labels, w)
            gradient_t_norm = np.linalg.norm(gradient_t)
            
            # eta_t = eta * |grad_t|
            eta_t = self.eta * gradient_t_norm
            w = w - eta_t * gradient_t

            self.loss.append(self.get_loss(self.training_data, self.training_labels, w))
            self.lr.append(eta_t)
            self.w_norm.append(gradient_t_norm)

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



