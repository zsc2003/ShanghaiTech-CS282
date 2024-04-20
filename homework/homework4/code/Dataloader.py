import pandas as pd

class DataLoader:
    def __init__(self, path):
        self.path = path
        self.data = self.load_data()

        self.process_data()

        # shuffle the data with fixed random state
        self.data_shuffled = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        self.training_data, self.training_label, self.testing_data, self.testing_label, self.val_data, self.val_label = self.split_data()

        # normalize the data
        training_mean = self.training_data.mean()
        training_std = self.training_data.std()
        testing_mean = self.testing_data.mean()
        testing_std = self.testing_data.std()
        val_mean = self.val_data.mean()
        val_std = self.val_data.std()

        self.training_data = (self.training_data - training_mean) / training_std
        self.testing_data = (self.testing_data - testing_mean) / testing_std
        self.val_data = (self.val_data - val_mean) / val_std

        
    def load_data(self):
        with open(self.path) as f:
            data = pd.read_csv(f)
        return data

    def process_data(self):
        self.data['Gender'] = self.data['Gender'].map({'Male' : 1, 'Female' : 0})

        if len(self.data['User ID'].unique()) == len(self.data):
            self.data = self.data.drop(columns=['User ID'])
        
        # seperate 'Age' into 6 groups, and use one-hot encoding
        # seperate as: 0-20, 20-26, 26-30, 30-40, 40-50, others
        self.data['Age'].hist(bins=20)

        self.data['Age'] = pd.cut(self.data['Age'], bins=[0, 20, 26, 30, 40, 50, 100], labels=[0, 1, 2, 3, 4, 5])

        self.data = pd.get_dummies(self.data, columns=['Age'], dtype=int)

        # seperate 'EstimatedSalary' into 8 groups, and use one-hot encoding
        # seperate as: 0-19500, 19500-40000, 40000-60000, 60000-80000, 80000-100000, 100000-130000, 130000-145000, others
        self.data['EstimatedSalary'].hist(bins=20)

        self.data['EstimatedSalary'] = pd.cut(self.data['EstimatedSalary'], bins=[0, 19500, 40000, 60000, 80000, 100000, 130000, 145000, 200000], labels=[0, 1, 2, 3, 4, 5, 6, 7])

        self.data = pd.get_dummies(self.data, columns=['EstimatedSalary'], dtype=int)

        # add a bias term
        self.data['bias'] = 1

        # modify Purchased to +1 and -1(0 -> -1)
        self.data['Purchased'] = self.data['Purchased'].map({0 : -1, 1 : 1})


    def split_data(self):
        training_rate = 0.6
        training_num = int(training_rate * len(self.data_shuffled))

        testing_rate = 0.2
        testing_num = int(testing_rate * len(self.data_shuffled))

        training_data = self.data_shuffled[:training_num]
        training_label = training_data['Purchased']
        training_data = training_data.drop(columns=['Purchased'])

        testing_data = self.data_shuffled[training_num:training_num+testing_num]
        testing_label = testing_data['Purchased']
        testing_data = testing_data.drop(columns=['Purchased'])

        val_data = self.data_shuffled[training_num+testing_num:]
        val_label = val_data['Purchased']
        val_data = val_data.drop(columns=['Purchased'])

        return training_data.to_numpy(), training_label.to_numpy().reshape(-1, 1), testing_data.to_numpy(), testing_label.to_numpy().reshape(-1, 1), val_data.to_numpy(), val_label.to_numpy().reshape(-1, 1)