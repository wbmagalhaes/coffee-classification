import tensorflow as tf


def create_model():
    model = tf.keras.models.Sequential()
    # Layer 1
    model.add(tf.keras.layers.BatchNormalization(input_shape=(64, 64, 3)))
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        bias_initializer='zeros',
        padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))

    # Layer 2
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(5, 5),
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        bias_initializer='zeros',
        padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))

    # Layer 3
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        bias_initializer='zeros',
        padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))

    # Layer 4
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        bias_initializer='zeros',
        padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))

    # Layer 5
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(
        filters=1024,
        kernel_size=(3, 3),
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        bias_initializer='zeros',
        padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))

    # Layer 6
    model.add(tf.keras.layers.Conv2D(
        filters=10,
        kernel_size=(3, 3),
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        bias_initializer='zeros',
        padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.05))

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Activation('softmax'))

    return model
