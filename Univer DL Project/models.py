import keras
import numpy as np
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, BatchNormalization
from tensorflow.keras.layers import Reshape, ReLU, MaxPooling1D, Add, Dropout, LayerNormalization, LSTM, GRU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import EfficientNetB0


def get_cnn_model(classes_num, input_size):
    if classes_num == 5:
        loss = losses.sparse_categorical_crossentropy
        last_activation = activations.softmax
    else:
        loss = losses.binary_crossentropy
        last_activation = activations.sigmoid
    

    inputs = Input(shape = (input_size, 1))
    x = Conv1D(32, kernel_size = 3, padding = "valid")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv1D(64, kernel_size = 3, padding = "valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    res_input = MaxPooling1D(pool_size = 2)(x)

    x = Conv1D(filters = 64, kernel_size = 3, padding = 'same')(res_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters = 64, kernel_size = 3, padding = 'same')(x)
    x = BatchNormalization()(x)
    residual_block = Add()([res_input, x])
    
    '''
    x = Dropout(rate = 0.2)(residual_block)

    x = Flatten()(x)
    x = Dense(50, activation = activations.relu)(x)
    last_layer = Dense(classes_num, activation = activations.softmax)(x)
    '''
    x = MaxPooling1D(pool_size = 2)(residual_block)
    x = Dropout(rate = 0.2)(x)
    x = Flatten()(x)
    x = Dense(50, activation = activations.relu)(x)
    last_layer = Dense(classes_num, activation = last_activation)(x)

    
    model = Model(inputs = inputs, outputs = last_layer)
    opt = optimizers.Adam(0.001)
    model.compile(optimizer = opt, loss = loss, metrics = ['acc'])
    model.summary()
    
    return model



def get_rnn_model(classes_num, input_size, rnn_start_dim = 90, time_steps = 18):
    '''
    rnn_start_dim - the size of the output vector after applying first Dense layer. 
    This vector is converted into the shape (18, 5) and then fed to the input of
    the LSTM layer.
    time_steps - number of time steps for LSTM cell.
    step_dim - the size of vector on each step of LSTM model. 
    '''
    
    if classes_num == 5:
        loss = losses.sparse_categorical_crossentropy
        last_activation = activations.softmax
    else:
        loss = losses.binary_crossentropy
        last_activation = activations.sigmoid
    
    inputs = Input(shape = (input_size, ))
    #inputs = Reshape((1, input_size))(inputs)
    step_dim = rnn_start_dim // time_steps
    x = Dense(rnn_start_dim)(inputs)
    x = Reshape((time_steps, step_dim))(x)
    
    x = LSTM(step_dim, return_sequences = True)(x)
    x = LayerNormalization()(x)
    
    x = LSTM(step_dim, return_sequences = True)(x)
    x = LayerNormalization()(x)

    x = GRU(step_dim, return_sequences = False)(x)
    last_layer = Dense(classes_num, activation = last_activation)(x)

    model = Model(inputs = inputs, outputs = last_layer)
    opt = optimizers.Adam(0.001)
    model.compile(optimizer = opt, loss = loss, metrics = ['acc'])
    model.summary()
    
    return model


def get_combined_model(classes_num, input_size):
    if classes_num == 5:
        loss = losses.sparse_categorical_crossentropy
        last_activation = activations.softmax
    else:
        loss = losses.binary_crossentropy
        last_activation = activations.sigmoid

    
    inputs = Input(shape = (input_size, 1))
    x = Conv1D(16, kernel_size = 3, padding = "valid")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size = 2)(x)
    
    x = Conv1D(32, kernel_size = 5, padding = "valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size = 2)(x)

    x = Conv1D(64, kernel_size = 1, padding = "valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size = 2)(x)
    x = Dropout(rate = 0.2)(x)

    x = LSTM(32, return_sequences = True)(x)
    x = LayerNormalization()(x)

    x = GRU(8, return_sequences = False)(x)
    x = LayerNormalization()(x)

    last_layer = Dense(classes_num, activation = last_activation)(x)
    model = Model(inputs = inputs, outputs = last_layer)
    opt = optimizers.Adam(0.0005)
    model.compile(optimizer = opt, loss = loss, metrics = ['acc'])
    model.summary()
    
    return model



