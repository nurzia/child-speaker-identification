"""
Taken from: https://github.com/Azure/DataScienceVM/blob/master/Tutorials/DeepLearningForAudio/Deep%20Learning%20for%20Audio%20Part%203%20-%20Training%20to%20Recognize%20Human%20Speech%20with%20Larger%20DNN.ipynb
"""

import tensorflow as tf
import keras

from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute, Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D, MaxPooling2D, Convolution1D, \
    MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import Input, merge, UpSampling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Multiply
from keras import optimizers


def build_toy_model(num_filters, num_classes, spec_len, feat_dim):
    model = Sequential()
    model.add(Flatten(input_shape=(spec_len, feat_dim)))
    model.add(Dense(num_filters, activation='relu'))
    model.add(Dense(num_classes, activation='softmax', name='output_layer'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_model_(num_classes, spec_len, feat_dim=64):
    n_time, n_freq = spec_len, feat_dim
    internal_reshape = max(12, min(64, n_freq))
    factor = max(1, min(4, round(pow(internal_reshape, 1. / 3))))

    # helper methods for DNN architecture
    def slice1(x):
        return x[:, :, :, 0:internal_reshape]

    def slice2(x):
        return x[:, :, :, internal_reshape:(2 * internal_reshape)]

    def slice1_output_shape(input_shape):
        return tuple([input_shape[0], input_shape[1], input_shape[2], internal_reshape])

    def slice2_output_shape(input_shape):
        return tuple([input_shape[0], input_shape[1], input_shape[2], internal_reshape])

    def block(input):
        cnn = Conv2D(2 * internal_reshape, (3, 3), padding="same", activation="linear", use_bias=False)(input)
        cnn = BatchNormalization(axis=-1)(cnn)

        cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
        cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

        cnn1 = Activation('linear')(cnn1)
        cnn2 = Activation('sigmoid')(cnn2)

        out = Multiply()([cnn1, cnn2])
        return out

    print('Internal CNN shapes will be' + internal_reshape
          + 'and' + 2 * internal_reshape + 'with factor' + factor)

    input_features = Input(shape=(n_time, n_freq), name='in_layer')
    a1 = Reshape((n_time, n_freq, 1))(input_features)

    cnn1 = a1
    while True:
        print (cnn1.shape)
        cnn1 = block(cnn1)
        cnn1 = block(cnn1)
        temp = MaxPooling2D(pool_size=(1, 2))(cnn1)
        if temp.shape[2] <= 16:
            break
        cnn1 = temp

    cnnout = Conv2D(factor * internal_reshape, (3, 3), padding="same", activation="relu", use_bias=True)(cnn1)
    cnnout = MaxPooling2D(pool_size=(1, factor))(cnnout)
    cnnout = Reshape((n_time, cnnout.shape.as_list()[3] * cnnout.shape.as_list()[2]))(cnnout)
    # Time step is downsampled to 30.

    rnnout = Bidirectional(GRU(internal_reshape, activation='linear', return_sequences=True, recurrent_dropout=0.5))(
        cnnout)
    rnnout_gate = Bidirectional(
        GRU(internal_reshape, activation='sigmoid', return_sequences=True, recurrent_dropout=0.5))(cnnout)
    out = Multiply(name='L')([rnnout, rnnout_gate])

    out = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(out)
    det = TimeDistributed(Dense(num_classes, activation='softmax'))(out)
    out = Multiply()([out, det])

    def outfunc(vects):
        x, y = vects
        # clip to avoid numerical underflow
        y = K.clip(y, 1e-7, 1.)
        y = K.sum(y, axis=1)
        x = K.sum(x, axis=1)
        return x / y

    out = Lambda(outfunc, output_shape=(num_classes,))([out, det])

    # print('out shape', out.shape)

    model = Model(input_features, out)
    model.summary()
    return model


def build_model(num_classes, spec_len, feat_dim=64):
    n_time, n_freq = spec_len, feat_dim

    input_features = Input(shape=(n_time, n_freq), name='in_layer')
    in_ = Reshape((n_time, n_freq, 1))(input_features)

    cnn = Conv2D(256, (3, 3), padding="same", activation="linear", use_bias=True)(in_)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = Activation('relu')(cnn)

    cnn = Conv2D(256, (3, 3), padding="same", activation="linear", use_bias=True)(cnn)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = Activation('relu')(cnn)
    # cnn = Dropout(0.5)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 1))(cnn)
    # cnn = Dropout(0.5)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 1))(cnn)

    cnn = Conv2D(128, (3, 3), padding="same", activation="linear", use_bias=True)(cnn)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = Activation('relu')(cnn)
    # cnn = Dropout(0.5)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 1))(cnn)
    cnn = Dropout(0.5)(cnn)
    # cnn = MaxPooling2D(pool_size=(2, 1))(cnn)

    cnn = Conv2D(64, (3, 3), padding="same", activation="linear", use_bias=True)(cnn)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = Activation('relu')(cnn)

    # cnn = Dropout(0.5)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 1))(cnn)

    # cnn = Dropout(0.5)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 1))(cnn)

    l = cnn.shape.as_list()
    cnnout = Reshape((l[1], l[2] * l[3]))(cnn)

    rnnout = Bidirectional(GRU(256, recurrent_dropout=.5, return_sequences=False), merge_mode='ave')(cnnout)

    out = Dense(num_classes, activation='softmax')(rnnout)

    model = Model(input_features, out)
    model.summary()

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


def build_diabolo(spec_len, feat_dim=64):
    n_time, n_freq = spec_len, feat_dim

    input_features = Input(shape=(n_time, n_freq), name='in_layer')
    in_ = Reshape((n_time, n_freq, 1))(input_features)

    cnn = Conv2D(16, (3, 3), padding="same", activation="relu", use_bias=True)(in_)
    cnn = MaxPooling2D(pool_size=(2, 2))(cnn)

    cnn = Conv2D(16, (3, 3), padding="same", activation="relu", use_bias=True)(cnn)
    encoded = MaxPooling2D(pool_size=(2, 2))(cnn)

    print(encoded.shape)

    ########################################################

    cnn = Conv2D(16, (3, 3), padding="same", activation="relu", use_bias=True)(encoded)
    cnn = UpSampling2D((2, 2))(cnn)

    cnn = Conv2D(16, (3, 3), padding="same", activation="relu", use_bias=True)(cnn)
    cnn = UpSampling2D((2, 2))(cnn)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(cnn)
    decoded = Reshape((n_time, n_freq))(decoded)

    print(decoded.shape)

    model = Model(input_features, decoded)
    model.summary()

    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam)
    return model
