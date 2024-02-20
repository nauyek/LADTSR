import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LSTM(object):

    def __init__(self, input_shape, units, epochs=100, name='LSTM_', if_train=True):
        self.input_shape = input_shape
        self.units = units
        self.name = name
        self.epochs = epochs
        self.if_train = if_train

    def get_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(self.units, input_shape=self.input_shape, return_sequences=True))
        model.add(tf.keras.layers.LSTM(self.units, return_sequences=False))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam())
        model.summary()
        return model

    def train(self, train_scaled):
        if self.if_train:
            # preprocess training data
            X_train, y_train = LSTM.train_generator(train_scaled, n_lags=self.input_shape[0])

            model = self.get_model()

            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                                  verbose=1, restore_best_weights=True)
            lr_red = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                          factor=0.2, patience=4, verbose=1,
                                                          min_lr=0.0000001, )
            callbacks = [es, lr_red]

            history = model.fit(X_train, y_train,
                                epochs=self.epochs,
                                validation_split=0.25,
                                batch_size=256,
                                verbose=1,
                                shuffle=False,
                                callbacks=callbacks)
            model.save(self.name)

    def pred(self, test_scaled):
        model = keras.models.load_model(self.name)
        # preprocess training data
        preds = model.predict(test_scaled)
        return preds

    @staticmethod
    def train_generator(dataset, n_lags=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - n_lags - 1):
            a = dataset.iloc[i:(i + n_lags)].to_numpy()
            dataX.append(a)
            dataY.append(dataset.iloc[i + n_lags].to_numpy())
        return (np.array(dataX), np.array(dataY))
