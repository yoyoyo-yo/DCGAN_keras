import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape, UpSampling2D, LeakyReLU, Activation, Conv2DTranspose

from keras.initializers import RandomNormal as RN, Constant

def G_model(Height, Width, channel=3):
    inputs = Input((100,))
    in_h = int(Height / 16)
    in_w = int(Width / 16)
    d_dim = 512
    x = Dense(in_h * in_w * d_dim, name='g_dense1',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs)
    x = Reshape((in_h, in_w, d_dim), input_shape=(d_dim * in_h * in_w,))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_dense1_bn')(x)
    # 1/8
    #x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(512, (5, 5), name='g_conv1', padding='same', strides=(2,2),
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = Conv2D(256, (5, 5), padding='same', name='g_conv1',
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv1_bn')(x)
    # 1/4
    #x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(256, (5, 5), name='g_conv2', padding='same', strides=(2,2),
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = Conv2D(128, (5, 5), padding='same', name='g_conv2',
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv2_bn')(x)
    # 1/2
    #x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(128, (5, 5), name='g_conv3', padding='same', strides=(2,2),
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = Conv2D(64, (5, 5), padding='same', name='g_conv3',
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv3_bn')(x)
    # 1/1
    #x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(channel, (5, 5), name='g_out', padding='same', strides=(2,2),
        kernel_initializer=RN(mean=0.0, stddev=0.02),  bias_initializer=Constant())(x)
    #x = Conv2D(channel, (5, 5), padding='same', activation='tanh', name='g_out',
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    x = Activation('tanh')(x)
    model = Model(inputs=inputs, outputs=x, name='G')
    return model

def D_model(Height, Width, channel=3):
    inputs = Input((Height, Width, channel))
    x = Conv2D(32, (5, 5), padding='same', strides=(2,2), name='d_conv1',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs)
    #x = InstanceNormalization()(x)
    #x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='d_conv1_bn')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, (5, 5), padding='same', strides=(2,2), name='d_conv2',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='d_conv2_bn')(x)
    #x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (5, 5), padding='same', strides=(2,2), name='d_conv3',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='d_conv3_bn')(x)
    #x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, (5, 5), padding='same', strides=(2,2), name='d_conv4',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    #x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='d_conv4_bn')(x)
    #x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    #x = Dense(2048, activation='relu', name='d_dense1',
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    x = Dense(1, activation='sigmoid', name='d_out',
        kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    model = Model(inputs=inputs, outputs=x, name='D')
    return model

def Combined_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model

from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf
class InstanceNormalization(Layer):
    def __init__(self, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)

        if K.backend() != "tensorflow":
            raise ValueError("only tf")

        if len(input_shape) == 2:
            mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
            return (inputs - mean) / K.sqrt(var + K.epsilon())
        elif len(input_shape)==4 and K.image_data_format() == "channels_last":
            mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
            return (inputs - mean) / K.sqrt(var + K.epsilon())
        else:
            raise ValueError("Not valid")

    def compute_output_shape(self, input_shape):
        return input_shape
