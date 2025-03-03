from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Activation, Multiply, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape, Add

def conv_block(x, filters, kernel=3, padding="same", activation=True):
    x = Conv2D(filters, kernel, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    return Activation("relu")(x) if activation else x

def spatial_attention(x):
    avg, max = GlobalAveragePooling2D()(x), GlobalMaxPooling2D()(x)
    attention = Dense(1, activation='sigmoid')(Concatenate()([avg, max]))
    return Multiply()([x, Reshape((1, 1, 1))(attention)])

def multires_block(x, filters):
    x1 = conv_block(x, int(filters*0.167))
    x2 = conv_block(x1, int(filters*0.333))
    x3 = conv_block(x2, int(filters*0.5))
    combined = BatchNormalization()(spatial_attention(Concatenate()([x1, x2, x3])))
    shortcut = conv_block(x, x1.shape[-1] + x2.shape[-1] + x3.shape[-1], 1, activation=False)
    return BatchNormalization()(Activation("relu")(combined + shortcut))

def res_path(x, filters, length):
    for _ in range(length):
        conv = conv_block(x, filters, activation=False)
        shortcut = conv_block(x, filters, 1, activation=False)
        attention = Dense(filters, activation='sigmoid')(GlobalAveragePooling2D()(conv))
        conv = Multiply()([conv, Reshape((1, 1, filters))(attention)])
        x = BatchNormalization()(Activation("relu")(Add()([conv, shortcut])))
    return x

def encoder_block(x, filters, length):
    x = multires_block(x, filters)
    return res_path(x, filters, length), MaxPooling2D(2)(x)

def decoder_block(x, skip, filters):
    x = Conv2DTranspose(filters, 2, 2, padding="same")(x)
    return multires_block(Concatenate()([x, skip]), filters)

def build_multiresunet(input_shape):
    inputs = Input(input_shape)
    s1, p1 = encoder_block(inputs, 32, 4)
    s2, p2 = encoder_block(p1, 64, 3)
    s3, p3 = encoder_block(p2, 128, 2)
    s4, p4 = encoder_block(p3, 256, 1)
    bridge = multires_block(p4, 512)
    d1 = decoder_block(bridge, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)
    return Model(inputs, Conv2D(1, 1, activation='sigmoid')(d4))
