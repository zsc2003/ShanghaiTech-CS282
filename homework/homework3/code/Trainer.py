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

        self.epoch = 5000
        self.eta = 0.2

        self.loss = []
        self.lr = []
        self.grad_norm = []
        self.testing_acc = []
        self.testing_loss = []
        self.func_evals_list = []
        self.grad_evals_list = []

        self.dimension = self.training_data.shape[1]

        self.func_evals = 0
        self.grad_evals = 0

    def get_loss(self, x, y, w):
        # logistic loss
        # X: (N, D), y: (N, 1), w: (D, 1) -> scalar
        self.func_evals += 1
        return np.mean(np.log(1 + np.exp(-y * np.dot(x, w))))

    def get_gradient(self, x, y, w):
        # logistic gradient
        # X: (N, D), y: (N, 1), w: (D, 1) -> (D, 1)
        self.grad_evals += 1
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
            self.testing_loss.append(self.get_loss(self.testing_data, self.testing_labels, w))
            self.lr.append(eta_t)
            self.grad_norm.append(gradient_t_norm)
            self.func_evals_list.append(self.func_evals)
            self.grad_evals_list.append(self.grad_evals)

            # calculate the accuracy on the testing data
            prediction = np.dot(self.testing_data, w)
            prediction = 1 / (1 + np.exp(-prediction))

            prediction[prediction > 0.5] = 1
            prediction[prediction <= 0.5] = -1
            accuracy = np.mean(prediction == self.testing_labels)
            self.testing_acc.append(accuracy)

    def print_info(self):

        print("====================================================================")
        print(" iter    loss        lr   |gradi|_2  test_acc  func_eval  grad_eval")
        iter_num = len(self.loss)
        for i in range(10, 0, -1):
            print(f" {iter_num - i + 1}  {self.loss[-i]:.6f}  {self.lr[-i]:.6f}  {self.grad_norm[-i]:.6f}  {self.testing_acc[-i]:.6f}     {self.func_evals_list[-i]}      {self.grad_evals_list[-i]}")
        print("====================================================================")

        plt.figure(1)
        plt.plot(self.loss)
        plt.title('Training Loss')
        plt.show()

        plt.figure(2)
        plt.plot(self.lr)
        plt.title('Learning Rate')
        plt.show()

        plt.figure(3)
        plt.plot(self.grad_norm)
        plt.title('Gradient Norm')
        plt.show()

        plt.figure(4)
        plt.plot(self.testing_acc)
        plt.title('Testing Accuracy')
        plt.show()

        plt.figure(5)
        plt.plot(self.testing_loss)
        plt.title('Testing Loss')
        plt.show()