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



def build_model_(num_classes, spec_len, feat_dim=64):

    n_time, n_freq = spec_len, feat_dim
    internal_reshape = max(12, min(64, n_freq))
    factor = max(1, min(4, round(pow(internal_reshape, 1. / 3))))

    # --------------------------------------------------------------------------------
    # functions inside build_model_:
    # helper methods for DNN architecture:

    def slice1(x):
        return x[:, :, :, 0:internal_reshape]

    def slice2(x):
        return x[:, :, :, internal_reshape:(2 * internal_reshape)]

    def slice_output_shape(input_shape):
        return tuple([input_shape[0], input_shape[1], input_shape[2], internal_reshape])

    def block(input):
        cnn = Conv2D(2 * internal_reshape, (3, 3),
                     padding="same", activation="linear",
                     use_bias=False)(input)
        cnn = BatchNormalization(axis=-1)(cnn)
        cnn1 = Lambda(slice1, output_shape=slice_output_shape)(cnn)
        cnn2 = Lambda(slice2, output_shape=slice_output_shape)(cnn)
        cnn1 = Activation('linear')(cnn1)
        cnn2 = Activation('sigmoid')(cnn2)
        out = Multiply()([cnn1, cnn2])
        return out

    # --------------------------------------------------------------------------------
    # output function

    def outfunc(vects):

        x, y = vects
        # clip to avoid numerical underflow
        y = K.clip(y, 1e-7, 1.)
        y = K.sum(y, axis=1)
        x = K.sum(x, axis=1)
        return x / y

    # --------------------------------------------------------------------------------
    # model building function

    input_features = Input(shape=(n_time, n_freq), name='in_layer')
    cnn1 = Reshape((n_time, n_freq, 1))(input_features)
    while True:
        cnn1 = block(cnn1)
        cnn1 = block(cnn1)
        temp = MaxPooling2D(pool_size=(1, 2))(cnn1)
        if temp.shape[2] <= 16:
            break
        cnn1 = temp

    cnnout = Conv2D(factor * internal_reshape, (3, 3), padding='same', activation='relu', use_bias=True)(cnn1)
    cnnout = MaxPooling2D(pool_size=(1, factor))(cnnout)
    cnnout = Reshape((n_time, cnnout.shape.as_list()[3] * cnnout.shape.as_list()[2]))(cnnout)

    rnnout = Bidirectional(GRU(internal_reshape, activation='linear', return_sequences=True, recurrent_dropout=0.5))(cnnout)
    rnnout_gate = Bidirectional(GRU(internal_reshape, activation='sigmoid', return_sequences=True, recurrent_dropout=0.5))(cnnout)

    out = Multiply(name='L')([rnnout, rnnout_gate])
    out = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='loc_layer_sig')(out)
    det = TimeDistributed(Dense(num_classes, activation='softmax'), name='loc_layer_sof')(out)
    out = Multiply(name='loc_layer_mult')([out, det])

    # out = Dense(num_classes, activation='softmax')(out)
    # out = TimeDistributed(Dense(num_classes, activation='softmax'), name='prova1')(out)
    # det = TimeDistributed(Dense(num_classes, activation='softmax'), name='prova2')(out)
    # out = Multiply(name='prova3')([out, det])

    out = Lambda(outfunc, output_shape=(num_classes,), name='out_layer')([out, det])

    model = Model(input_features, out)
    model.summary()
    return model
