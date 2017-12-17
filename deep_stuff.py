import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input

def deep_mlp(input_dims):
    model = Sequential()

    model.add(Dense(64, activation='sigmoid', input_dim=input_dims))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(8, activation='linear'))

    return model