import pandas as pd
import seaborn
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, path):
        self.path = path
        self.data = self.load_data()

        self.process_data()

        # shuffle the data with fixed random state
        self.data_shuffled = self.data.sample(frac=1, random_state=0).reset_index(drop=True)
        self.training_data, self.training_label, self.testing_data, self.testing_label = self.split_data()
        
    def load_data(self):
        with open(self.path) as f:
            data = pd.read_csv(f)
        return data

    def process_data(self):
        self.data['Gender'] = self.data['Gender'].map({'Male' : 1, 'Female' : 0})

        if len(self.data['User ID'].unique()) == len(self.data):
            print('All elements in the \'User ID\' column is unique')
            self.data = self.data.drop(columns=['User ID'])
        
        # seperate 'Age' into 6 groups, and use one-hot encoding
        # seperate as: 0-20, 20-26, 26-30, 30-40, 40-50, others
        self.data['Age'].hist(bins=20)
        plt.xlabel('Age'), plt.ylabel('Number'), plt.title('distribution of Age')
        plt.show()

        self.data['Age'] = pd.cut(self.data['Age'], bins=[0, 20, 26, 30, 40, 50, 100], labels=[0, 1, 2, 3, 4, 5])
        seaborn.countplot(x='Purchased', hue='Age', data=self.data)
        plt.title('sepreate Age into 6 groups')
        plt.show()

        self.data = pd.get_dummies(self.data, columns=['Age'], dtype=int)

        # seperate 'EstimatedSalary' into 8 groups, and use one-hot encoding
        # seperate as: 0-19500, 19500-40000, 40000-60000, 60000-80000, 80000-100000, 100000-130000, 130000-145000, others
        self.data['EstimatedSalary'].hist(bins=20)
        plt.xlabel('Estimated Salary'), plt.ylabel('Number'), plt.title('distribution of Estimated Salary')
        plt.show()

        self.data['EstimatedSalary'] = pd.cut(self.data['EstimatedSalary'], bins=[0, 19500, 40000, 60000, 80000, 100000, 130000, 145000, 200000], labels=[0, 1, 2, 3, 4, 5, 6, 7])
        seaborn.countplot(x='Purchased', hue='EstimatedSalary', data=self.data)
        plt.title('sepreate Estimated Salary into 8 groups')
        plt.show()

        self.data = pd.get_dummies(self.data, columns=['EstimatedSalary'], dtype=int)

    def split_data(self):
        training_rate = 0.7
        training_num = int(training_rate * len(self.data_shuffled))

        training_data = self.data_shuffled[:training_num]
        training_label = training_data['Purchased']
        training_data = training_data.drop(columns=['Purchased'])

        testing_data = self.data_shuffled[training_num:]
        testing_label = testing_data['Purchased']
        testing_data = testing_data.drop(columns=['Purchased'])

        return training_data.to_numpy(), training_label.to_numpy().reshape(-1, 1), testing_data.to_numpy(), testing_label.to_numpy().reshape(-1, 1)