import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau


def train(model, save_model_file_path: str, epochs_num: int, x_train: np.ndarray, y_train: np.ndarray):
    checkpoint = ModelCheckpoint(save_model_file_path, monitor = 'val_acc', verbose = 1, 
                                 save_best_only = True, mode = 'max')
    early = EarlyStopping(monitor = "val_acc", mode = "max", patience = 5, verbose = 1)
    redonplat = ReduceLROnPlateau(monitor = "val_acc", mode = "max", patience = 3, verbose = 2)
    callbacks_list = [checkpoint, early, redonplat]  # early

    model.fit(x_train, y_train, epochs = epochs_num, verbose = 2, 
              callbacks = callbacks_list, validation_split = 0.1)
    model.load_weights(save_model_file_path)



