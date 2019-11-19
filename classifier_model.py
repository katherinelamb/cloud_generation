import numpy as np
import os
import numpy as np
import tensorflow as tf


def classifier(input_size=(128, 128, 3), pretrained_weights=None, summary=False):
    he_initializer = tf.keras.initializers.he_normal(seed=0)

    inputs = tf.keras.layers.Input(input_size)
    conv1 = tf.keras.layers.Conv2D(64,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer=he_initializer)(inputs)
    conv1 = tf.keras.layers.Conv2D(64,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer=he_initializer)(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer=he_initializer)(pool1)
    conv2 = tf.keras.layers.Conv2D(128,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer=he_initializer)(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer=he_initializer)(pool2)
    conv3 = tf.keras.layers.Conv2D(256,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer=he_initializer)(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(512,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer=he_initializer)(pool3)
    conv4 = tf.keras.layers.Conv2D(512,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer=he_initializer)(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tf.keras.layers.Conv2D(1024,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer=he_initializer)(pool4)
    conv5 = tf.keras.layers.Conv2D(1024,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer=he_initializer)(conv5)
    gpool = tf.keras.layers.GlobalMaxPooling2D()(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(gpool)
    flat = tf.keras.layers.Flatten()(drop5)
    dense1 = tf.keras.layers.Dense(4096, activation='relu', kernel_initializer=he_initializer)(flat)
    drop6 = tf.keras.layers.Dropout(0.5)(dense1)
    dense2 = tf.keras.layers.Dense(4096, activation='relu', kernel_initializer=he_initializer)(drop6)
    drop7 = tf.keras.layers.Dropout(0.5)(dense2)
    output = tf.keras.layers.Dense(4, activation='softmax')(drop7)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    if summary:
        model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
