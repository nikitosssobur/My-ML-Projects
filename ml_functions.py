import numpy as np
import math


'''
------------------------------ Activation functions --------------------------------
'''

def softmax(vector):    return np.exp(vector) / np.sum(vector)


def ReLU(x, deriv = False):
    return max(0, x) if not deriv else (0 if x <= 0 else 1)


def binary_step_func(x): 
    return 1 if x >= 0 else 0


def linear(x, a = 1, b = 0, deriv = False):
    return a * x + b if not deriv else a


def sigmoid(x, a = 1, b = 1, deriv = False):
    return a / (1 + np.exp(-b * x)) if not deriv else sigmoid(x) * (1 - sigmoid(x))


def tanh(x, a = 1, b = 1, c = 1, d = 1):
    return (np.exp(a * x) - np.exp(-b * x)) / (np.exp(c * x) + np.exp(-d * x))

    
def leaky_relu(x, a = 0.1):  return max(a * x, x)


def elu(x, a):  return x if x >= 0 else a * (np.exp(x) - 1)


def swish(x, a = 1, b = 1):
    return x * sigmoid(x) if a == 1 and b == 1 else x * sigmoid(x, a, b)


def selu(x, a1 = 0.1, a2 = 0.1):
    return a1 * x if x >= 0 else a1 * a2 * (np.exp(x) - 1)


def gompertz(x, a = 1, b = -1, c = -1):  return a * np.exp(b * np.exp(c * x))

'''
----------------------------------- Metrics and losses ----------------------------------
'''


def check_problems_decorator(func):
    def wrapper(*args, **kwargs):
        if (len(args[0]) != len(args[1])) or np.shape(args[0]) != np.shape(args[1]):
            raise ValueError('Arguments have different shapes!')  
        else: return func(*args, **kwargs)  
    return wrapper


@check_problems_decorator
def MSE(y_pred, y_target):
    return sum([(y1 - y2)**2 for y1, y2 in zip(y_pred, y_target)]) / len(y_pred)


@check_problems_decorator
def MAE(y_pred, y_target):
    return sum([abs(y1 - y2) for y1, y2 in zip(y_pred, y_target)]) / len(y_pred)


@check_problems_decorator
def r_squared(y_pred, y_target):
    target_avg = sum(y_target) / len(y_target) 
    return 1 - (sum([(y1 - y2)**2 for y1, y2 in zip(y_pred, y_target)]) / sum([(target_avg - y)**2 for y in y_target]))


@check_problems_decorator  
def MAPE(y_pred, y_target):
    return sum([100 * abs(y2 - y1)/abs(y2) for y1, y2 in zip(y_pred, y_target)]) / len(y_pred)


@check_problems_decorator
def MSLE(y_pred, y_target, base = np.e):
    if base == np.e:
        return sum([(np.log(y1 + 1) - np.log(y2 + 1))**2 for y1,y2 in zip(y_pred, y_target)]) / len(y_pred)
    else:
        return sum([(math.log(y1 + 1, base) - math.log(y2 + 1, base))**2 for y1, y2 in zip(y_pred, y_target)]) / len(y_pred)


@check_problems_decorator
def RMSLE(y_pred, y_target, base = np.e):
    return math.sqrt(MSLE(y_pred, y_target, base))


#def binary_cross_entrophy():
    

def huber_loss(x, c = 1):
    return x ** 2 if abs(x) < c else 2 * c * abs(x) - c ** 2


#def quantile_loss():

