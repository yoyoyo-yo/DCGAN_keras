import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape, UpSampling2D, concatenate


def G_model(Height, Width, channel=3):
    inputs_z = Input((100,), name='Z')
    in_h = int(Height / 4)
    in_w = int(Width / 4)
    x = Dense(in_h * in_w * 128, activation='tanh', name='g_dense1')(inputs_z)
    x = BatchNormalization()(x)
    x = Reshape((in_h, in_w, 128), input_shape=(128 * in_h * in_w,))(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (5, 5), padding='same', activation='tanh', name='g_conv1')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(channel, (5, 5), padding='same', activation='tanh', name='g_out')(x)
    model = Model(inputs=[inputs_z], outputs=[x], name='G')
    return model

def D_model(Height, Width, channel=3):
    inputs_x = Input((Height, Width, channel), name='X')
    x = Conv2D(64, (5, 5), padding='same', activation='tanh', name='d_conv1')(inputs_x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (5, 5), padding='same', activation='tanh', name='d_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='d_dense1')(x)
    x = Dense(1, activation='sigmoid', name='d_out')(x)
    model = Model(inputs=[inputs_x], outputs=[x], name='D')
    return model

def Combined_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model
