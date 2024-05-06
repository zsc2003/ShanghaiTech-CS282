import matplotlib.pyplot as plt
import numpy as np
import tqdm
from typing import List

class Trainer:
    from Dataloader import DataLoader
    def __init__(self, data: DataLoader, L1_lambda: List[float], L2_lambda: List[float]):
        self.training_data = data.training_data
        self.training_labels = data.training_label

        self.testing_data = data.testing_data
        self.testing_labels = data.testing_label

        self.column_names = data.column_names
        self.dimension = self.training_data.shape[1]
   
        self.epoch = 10000
        self.eta = 0.2

        self.L1_lambda = L1_lambda
        self.L2_lambda = L2_lambda
        self.L1_params = []
        self.L2_params = []

    def get_loss(self, x, y, w, regularization_type='None', lam=0):
        # logistic loss
        # X: (N, D), y: (N, 1), w: (D, 1) -> scalar
        if regularization_type == 'L1':
            return np.mean(np.log(1 + np.exp(-y * np.dot(x, w)))) + lam * np.linalg.norm(w, ord=1)
        elif regularization_type == 'L2':
            return np.mean(np.log(1 + np.exp(-y * np.dot(x, w)))) + lam * np.linalg.norm(w, ord=2) ** 2
        else:
            return np.mean(np.log(1 + np.exp(-y * np.dot(x, w))))

    def get_gradient(self, x, y, w, regularization_type='None', lam=0):
        # logistic gradient
        # X: (N, D), y: (N, 1), w: (D, 1) -> (D, 1)
        if regularization_type == 'L1':
            return -np.mean(y * x / (1 + np.exp(y * np.dot(x, w))), axis=0).reshape(-1, 1) + lam * np.sign(w)
        elif regularization_type == 'L2':
            return -np.mean(y * x / (1 + np.exp(y * np.dot(x, w))), axis=0).reshape(-1, 1) + 2 * lam * w
        else:
            return -np.mean(y * x / (1 + np.exp(y * np.dot(x, w))), axis=0).reshape(-1, 1)
            
    def train(self, regularization_type='None', lam=0):
        w = np.zeros(self.dimension).reshape(-1, 1)
        best_w = w
        for _ in tqdm.tqdm(range(self.epoch)):
            gradient_t = self.get_gradient(self.training_data, self.training_labels, w, regularization_type, lam)
            gradient_t_norm = np.linalg.norm(gradient_t)
            
            # eta_t = eta * |grad_t|
            eta_t = self.eta * gradient_t_norm
            w = w - eta_t * gradient_t
            if self.get_loss(self.testing_data, self.testing_labels, w, regularization_type, lam) < self.get_loss(self.testing_data, self.testing_labels, best_w, regularization_type, lam):
                best_w = w

        # save the parameters
        if regularization_type == 'L1':
            self.L1_params.append(best_w.reshape(-1,))
        elif regularization_type == 'L2':
            self.L2_params.append(best_w)
        else:
            pass

    def print_info(self):
        # show the path plot of L1 and L2 regularization

        # show each feature with its name and its path
        plt.figure(1)
        weight = [[]] * self.dimension
        for i in range(self.dimension):
            weight[i] = [self.L1_params[j][i] for j in range(len(self.L1_params))]

        for i, L1_param in enumerate(weight):
            plt.plot(L1_param, label=self.column_names[i])
            plt.legend()

        # mapping the x-axis to lambda
        x_axis = [i for i in range(len(self.L1_lambda))]
        plt.xticks(x_axis, self.L1_lambda)
        plt.title('L1 regularization')
        plt.xlabel('lambda')
        plt.ylabel('weights')
        plt.show()


        # show each feature with its name and its path
        plt.figure(2)
        weight = [[]] * self.dimension
        for i in range(self.dimension):
            weight[i] = [self.L2_params[j][i] for j in range(len(self.L2_params))]

        for i, L2_param in enumerate(weight):
            plt.plot(L2_param, label=self.column_names[i])
            plt.legend()

        # mapping the x-axis to lambda
        x_axis = [i for i in range(len(self.L2_lambda))]
        plt.xticks(x_axis, self.L2_lambda)
        plt.title('L2 regularization')
        plt.xlabel('lambda')
        plt.ylabel('weights')
        plt.show()