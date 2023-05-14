import os 
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten, Lambda

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Model should be fined tune for dynamic frames of lips
def load_model(width, height):
    oldModel = Sequential()
    oldModel.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    oldModel.add(Activation('relu'))
    oldModel.add(MaxPool3D((1,2,2), padding='same'))

    oldModel.add(Conv3D(256, 3, padding='same'))
    oldModel.add(Activation('relu'))
    oldModel.add(MaxPool3D((1,2,2), padding='same'))

    oldModel.add(Conv3D(75, 3, padding='same'))
    oldModel.add(Activation('relu'))
    oldModel.add(MaxPool3D((1,2,2), padding='same'))

    oldModel.add(TimeDistributed(Flatten()))

    oldModel.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    oldModel.add(Dropout(.5))

    oldModel.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    oldModel.add(Dropout(.5))

    oldModel.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    oldModel.load_weights(os.path.join('..','models','checkpoint'))

    # newInput = Input(batch_shape=(75, width, height,3))
    # newOutputs = oldModel(newInput)
    # newModel = Model(newInput,newOutputs)
    
    newModel = Sequential()
    # newModel.add(Lambda(lambda x: tf.expand_dims(newModel.output, axis=-1)))
    newModel.add(Conv3D(128, 3, input_shape=(75, width, height, 3), padding='same'))
    newModel.add(Activation('relu'))
    newModel.add(MaxPool3D((1,2,2), padding='same'))

    newModel.add(Conv3D(256, 3, padding='same'))
    newModel.add(Activation('relu'))
    newModel.add(MaxPool3D((1,2,2), padding='same'))

    newModel.add(Conv3D(75, 3, padding='same'))
    newModel.add(Activation('relu'))
    newModel.add(MaxPool3D((1,2,2), padding='same'))

    newModel.add(TimeDistributed(Flatten()))

    newModel.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    newModel.add(Dropout(.5))

    newModel.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    newModel.add(Dropout(.5))

    newModel.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))
    newModel.set_weights(oldModel.get_weights())
    return newModel