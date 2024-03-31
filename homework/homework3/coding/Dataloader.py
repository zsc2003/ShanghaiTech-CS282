import pandas as pd


class DataLoader:
    def __init__(self, path):
        self.path = path
        self.data = self.load_data()

        self.process_data()
        
        self.data_shuffled = self.data.sample(frac=1, random_state=0).reset_index(drop=True)
        self.training_data, self.valid_data, self.testing_data = self.split_data()
   
        self.training_data = self.training_data.to_numpy()
        self.valid_data = self.valid_data.to_numpy()
        self.testing_data = self.testing_data.to_numpy()

    def load_data(self):
        with open(self.path) as f:
            data = pd.read_csv(f)
        return data

    def process_data(self):
        self.data['Gender'] = self.data['Gender'].map({'Male' : 1, 'Female' : 0})
        if len(self.data['User ID'].unique()) == len(self.data):
            print('All elements in the \'User ID\' column is unique')
            self.data = self.data.drop(columns=['User ID'])
    
    def split_data(self):
        training_rate = 0.7
        valid_rate = 0.15
        testing_rate = 0.15

        training_num = int(training_rate * len(self.data_shuffled))
        valid_num = int(valid_rate * len(self.data_shuffled))
        testing_num = len(self.data_shuffled) - training_num - valid_num

        training_data = self.data_shuffled[:training_num]
        valid_data = self.data_shuffled[training_num:training_num+valid_num]
        testing_data = self.data_shuffled[training_num+valid_num:]

        return training_data, valid_data, testing_data