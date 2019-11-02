import tensorflow as tf

kernel_initializer = 'he_normal'
kernel_regularizer = tf.keras.regularizers.l2(0.01)
bias_initializer = tf.keras.initializers.Constant(value=0.1)
leaky_relu_alpha = 0.2
drop_rate = 0.25


def create_model():
    image_input = tf.keras.Input(shape=(64, 64, 3), name='img_input')

    # Layer 1
    x = tf.keras.layers.BatchNormalization()(image_input)
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Dropout(rate=drop_rate)(x)

    # Layer 2
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(5, 5),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Dropout(rate=drop_rate)(x)

    # Layer 3
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Dropout(rate=drop_rate)(x)

    # Layer 4
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Dropout(rate=drop_rate)(x)

    # # Layer 5
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        filters=1024,
        kernel_size=(3, 3),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Dropout(rate=drop_rate)(x)

    # Layer 6
    x = tf.keras.layers.Conv2D(
        filters=10,
        kernel_size=(3, 3),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(x)

    logits = tf.keras.layers.GlobalAveragePooling2D(name='logits')(x)
    classes = tf.keras.layers.Activation('softmax', name='classes')(logits)

    model = tf.keras.Model(inputs=[image_input], outputs=[logits, classes])
    return model
