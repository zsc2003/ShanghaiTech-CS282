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

        self.val_data = data.val_data
        self.val_labels = data.val_label

        self.epoch = 50000

        self.train_loss = []
        self.val_acc = []
        self.val_loss = []

        self.dimension = self.training_data.shape[1]

        self.func_evals = 0
        self.grad_evals = 0

        self.etas = []
        self.w = []

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
    
    def train(self, eta):
        self.train_loss.append([])
        self.val_acc.append([])
        self.val_loss.append([])
        self.etas.append(eta)
        self.w.append([])
        
        w = np.zeros(self.dimension).reshape(-1, 1)
        for _ in tqdm.tqdm(range(self.epoch)):
            # randomly choose a sample from training data, and get its label
            idx = np.random.randint(0, self.training_data.shape[0])
            x = self.training_data[idx].reshape(1, -1)
            y = self.training_labels[idx].reshape(1, 1)
            
            gradient_t = self.get_gradient(x, y, w)
            
            w = w - eta * gradient_t

            self.w[-1].append(w)
            self.train_loss[-1].append(self.get_loss(self.training_data, self.training_labels, w))
            self.val_loss[-1].append(self.get_loss(self.val_data, self.val_labels, w))

            # calculate the accuracy on the validation data
            prediction = np.dot(self.val_data, w)
            prediction = 1 / (1 + np.exp(-prediction))

            prediction[prediction > 0.5] = 1
            prediction[prediction <= 0.5] = -1
            accuracy = np.mean(prediction == self.val_labels)
            self.val_acc[-1].append(accuracy)

    def inference(self):
        # take the min accuracy from the last 1000 epochs as the best accuracy
        val_acc = np.array(self.val_acc)
        val_acc = np.mean(val_acc[:, -1000:], axis=1)

        best_idx = np.argmax(val_acc)
        self.best_eta = self.etas[best_idx]

        self.testing_acc = []
        self.testing_loss = []

        w = self.w[best_idx]
        for t in tqdm.tqdm(range(self.epoch)):
            self.testing_loss.append(self.get_loss(self.testing_data, self.testing_labels, w[t]))
            
            # calculate the accuracy on the validation data
            prediction = np.dot(self.testing_data, w[t])
            prediction = 1 / (1 + np.exp(-prediction))

            prediction[prediction > 0.5] = 1
            prediction[prediction <= 0.5] = -1
            accuracy = np.mean(prediction == self.testing_labels)
            self.testing_acc.append(accuracy)
        
        print(f'Best eta = {self.best_eta}')
        print(f'Testing accuracy = {self.testing_acc[-1]}')

    def print_info(self):
        # clear the plot
        plt.close('all')

        # plot the training loss for each eta
        plt.figure(1)
        for i, eta in enumerate(self.etas):
            plt.plot(self.train_loss[i], label=f'eta = {eta}')
        plt.title('Training Loss')
        plt.legend()
        plt.show()

        plt.figure(2)
        for i, eta in enumerate(self.etas):
            plt.plot(self.train_loss[i][-1000:], label=f'eta = {eta}')
        plt.title('Training Loss convergence part')
        plt.legend()
        plt.show()

        plt.figure(3)
        for i, eta in enumerate(self.etas):
            plt.plot(self.val_acc[i], label=f'eta = {eta}')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.show()

        plt.figure(4)
        for i, eta in enumerate(self.etas):
            plt.plot(self.val_loss[i], label=f'eta = {eta}')
        plt.title('Validation Loss')
        plt.legend()
        plt.show()

        # best eta
        plt.figure(5)
        plt.plot(self.testing_acc)
        plt.title(f'Testing Accuracy with best eta = {self.best_eta}')
        plt.show()

        plt.figure(6)
        plt.plot(self.testing_loss)
        plt.title(f'Testing Loss with best eta = {self.best_eta}')
        plt.show()