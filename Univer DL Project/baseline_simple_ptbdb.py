import pandas as pd
import numpy as np

import tensorflow
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Convolution1D, Flatten
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, average_precision_score
from sklearn.model_selection import train_test_split

# ---------- INPUT ----------
df_1 = pd.read_csv("data/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("data/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])


Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

# ---------- MODEL ----------
def get_vanillaCNN():
    nclass = 1
    inp = Input(shape=(187, 1))
    cnn_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    cnn_1 = Flatten()(cnn_1)
    
    dense_1 = Dense(nclass, activation=activations.sigmoid, name="dense_1")(cnn_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    
    return model

model = get_vanillaCNN()
file_path = "vanillaCNN_ptbdb.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

# ---------- TRAIN ----------
model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
model.load_weights(file_path)

# ---------- EVAL ----------
pred_test = model.predict(X_test)
pred_test = (pred_test>0.5).astype(np.int8)

f1 = f1_score(Y_test, pred_test)

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)

auroc = roc_auc_score(Y_test, pred_test)

print("Test AUROC score : %s "% auroc)

auprc = average_precision_score(Y_test, pred_test)

print("Test average_precision_score AUPRC score : %s "% auprc)