import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#--------------MITBIH Data processing --------------
'''
df_train = pd.read_csv("data/mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("data/mitbih_test.csv", header=None)

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
'''


#----------------Ptbdb--------------
'''
df_1 = pd.read_csv("data/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("data/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])


Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
'''

class Datasets:

    def __init__(self, mitbih_train_path, mitbih_test_path, ptbdb_abnormal_path, ptbdb_normal_path):
        self.mitbih_train, self.mitbih_test = mitbih_train_path, mitbih_test_path
        self.ptbdb_abnormal, self.ptbdb_normal = ptbdb_abnormal_path, ptbdb_normal_path
        self.mitbih_train_data = self.read_data(self.mitbih_train)
        self.mitbih_test_data = self.read_data(self.mitbih_test)
        self.ptbdb_abnormal_data = self.read_data(self.ptbdb_abnormal)
        self.ptbdb_normal_data = self.read_data(self.ptbdb_normal)


    def read_data(self, path):
        return pd.read_csv(path, header = None)
    

    def get_inputs_outputs_data(self, dataset):
        Y = np.array(dataset[187].values).astype(np.int8)
        X = np.array(dataset[list(range(187))].values)[..., np.newaxis]
        return X, Y


    def dataset_len(self, dataset):
        return len(dataset)


    def get_mitbih_data(self):
        df_train = self.mitbih_train_data.sample(frac = 1)     
        X_train, Y_train = self.get_inputs_outputs_data(df_train) 
        X_test, Y_test = self.get_inputs_outputs_data(self.mitbih_test_data)
        return X_train, Y_train, X_test, Y_test


    def get_ptbdb_data(self):
        df = pd.concat([self.ptbdb_normal_data, self.ptbdb_abnormal_data])
        df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1337, stratify = df[187])
        X_train, Y_train = self.get_inputs_outputs_data(df_train)
        X_test, Y_test = self.get_inputs_outputs_data(df_test)  
        return X_train, Y_train, X_test, Y_test

    
    def calculate_labels_distribution(self, dataset):
        labels = dataset.iloc[:, -1]
        unique_labels = pd.unique(labels)
        data_len = self.dataset_len(labels)
        return {int(label): round(len(labels[labels == label]) / data_len, 4) for label in unique_labels} 


'''
datasets = Datasets("data/mitbih_train.csv", "data/mitbih_test.csv", 
                    "data/ptbdb_abnormal.csv",  "data/ptbdb_normal.csv")

x_train1, y_train1, x_test1, y_test1 = datasets.get_mitbih_data()

x_train2, y_train2, x_test2, y_test2 = datasets.get_ptbdb_data()

print(datasets.calculate_labels_distribution(datasets.mitbih_train_data))

print(x_train1.shape, y_train1.shape)
print(x_test1.shape, y_test1.shape)
print(x_train2.shape, y_train2.shape)
print(x_test2.shape, y_test2.shape)
'''