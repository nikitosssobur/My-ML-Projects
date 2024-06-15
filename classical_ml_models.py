import numpy as np
import matplotlib.pyplot as plt
import math



class LinearRegression:
    def __init__(self): self.slope, self.intercept = 0, 0
        
    def train(self, xi_data, yi_data):
        if np.shape(xi_data) == np.shape(yi_data):
            self.xi_data, self.yi_data, n = xi_data, yi_data, len(xi_data)
            sum_xy, sum_x, sum_y = np.sum(np.dot(xi_data, yi_data)), np.sum(xi_data), np.sum(yi_data)
            sum_x_sq = np.sum(np.dot(xi_data, xi_data))
            self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - (sum_x) ** 2)
            self.intercept = (sum_y - self.slope * sum_x) / n
        else: print("Error! Shapes of X data array and Y data array aren't equivalent!")

    def predict(self, x): return x * self.slope + self.intercept

    def get_coeffs(self): return {'slope': self.slope, 'intercept': self.intercept}



class MultiLinearRegression:
    #Multilinear regression model

    def __init__(self):
        self.teta, self.x_matrix = None, None
        self.y_estim, self.var = None, None
    

    def __set_data(self, x_data, y_data):
        if y_data is None:
            self.y_data = [row[0] for row in x_data]
            self.x_data = [row[1:] for row in x_data]
        else:
            self.x_data, self.y_data = x_data, y_data

    
    __generate_x_matrix = lambda self: np.array([[1] + row for row in self.x_data], dtype=float)
    

    def train(self, x_data, y_data = None):
        self.__set_data(x_data, y_data)
        self.x_matrix = self.__generate_x_matrix()
        transp_x = np.transpose(self.x_matrix) 
        self.teta = np.dot(np.linalg.inv(np.dot(transp_x, self.x_matrix)), np.dot(transp_x, y_data))


    def get_coeffs(self): return self.teta


    def predict(self, x_input): return np.dot(np.concatenate(([1], x_input)), self.teta)
    

    def __calculate_estimation(self): self.y_estim = np.dot(self.x_matrix, self.teta)    


    def get_estimation(self): 
        if self.y_estim is None: self.__calculate_estimation()
        return self.y_estim


    def get_predictions(self, x_inputs):
        return np.array([self.predict(x_input) for x_input in x_inputs])
    
    
    def __var_estimation(self):
        n, p = len(self.x_data), len(self.x_data[0])
        self.var = sum([(self.y_data[i] - self.y_estim[i]) ** 2  for i in range(n)]) / (n - p - 1)


    def get_var(self):
        if self.var is None: self.__var_estimation()
        return self.var


class LogisticRegression:
    '''
    Binary classification logistic regression model
    '''

    def __init__(self, in_features):
        self.in_features = in_features
        self.weights = np.random.rand(in_features + 1)


    def set_train_data(self, inputs, targets):
        self.inputs = inputs  #matrix of vectors-objects x of size in_features from training sample
        self.targets = targets #vector of answers (class label) for each input x 


    def logit_func(self, arg):
        return 1 / (1 + np.exp(-arg))


    def train(self, lr, epoch_num):
        self.lr = lr
        for _ in range(epoch_num):
            for xi, yi in zip(self.inputs, self.targets):
                y_pred = self.__forward(xi, yi)    
                self.__backward(np.concatenate(xi, [1]), yi, y_pred)


    def __forward(self, x, true_output):
        return self.logit_func(self.margin(x) * true_output)


    def __backward(self, x, y_true, y_pred):
        self.weights += self.lr * y_true * y_pred * x


    def margin(self, x):
        if len(x) == self.in_features:
            return np.dot(self.weights, np.concatenate(x, [1]))
    

    def predict(self, x): 
        return self.logit_func(self.margin(x))
    

    def predict_class(self, x, classes = [1, -1]):
        return classes[0] if self.predict(x) > 0.5 else classes[1]



class MultiClassLogisticRegression:
    def __init__(self, features_num, classes_num):
        self.features_num = features_num
        self.classes_num = classes_num
        self.weights = np.random.rand(self.classes_num, self.features_num + 1)
        self.classes = np.arange(self.classes_num)


    def set_train_data(self, inputs, targets):
        self.inputs = inputs 
        self.targets = targets


    def softmax(self, vector_arg):
        return np.exp(vector_arg) / np.sum(np.exp(vector_arg))

    '''
    def train(self, lr, epoch_num):
        self.lr = lr
        for _ in range(epoch_num):
            for xi, yi in zip(self.inputs, self.targets):
                y_pred = self.__forward(xi, yi)    
                self.__backward(np.concatenate(xi, [1]), yi, y_pred)


    def __forward(self, x, true_output):
        return self.logit_func(self.margin(x) * true_output)


    def __backward(self, x, y_true, y_pred):
        self.weights += self.lr * y_true * y_pred * x
    '''

    def predict(self, x): 
        return self.softmax(np.dot(self.weights, x))
    

    def predict_class(self, x):
        return np.argmax(self.predict(x))
    


