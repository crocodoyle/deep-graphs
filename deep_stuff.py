import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise
from keras.regularizers import l1, l2, l1_l2
import keras.backend as K

from keras.callbacks import LearningRateScheduler

def deep_mlp(input_dims, output_dims):
    model = Sequential()

    model.add(Dense(16, activation='sigmoid', input_dim=input_dims))
    model.add(Dropout(0.25))
    model.add(GaussianNoise(0.01))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='sigmoid'))
    # model.add(GaussianNoise(0.01))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='sigmoid'))
    # model.add(GaussianNoise(0.01))
    # model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    model.add(Dense(16, activation='sigmoid'))
    # model.add(GaussianNoise(0.02))
    # model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    # model.add(Dense(64, activation='sigmoid'))
    # model.add(GaussianNoise(0.01))
    # model.add(Dropout(0.5))
    # model.add(BatchNormalization())

    model.add(Dense(output_dims, activation='linear'))

    return model

def lr_scheduler(model):
    # reduce learning rate by factor of 10 every 100 epochs
    def schedule(epoch):
        new_lr = K.get_value(model.optimizer.lr)

        if epoch % 100 == 0:
            new_lr = new_lr / 2

        return new_lr

    scheduler = LearningRateScheduler(schedule)
    return scheduler


