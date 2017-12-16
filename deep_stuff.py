import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input

def deep_mlp(input_dims):
    model = Sequential()

    model.add(Dense(256, activation='relu', input_dim=input_dims))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(8, activation='linear'))

    return model