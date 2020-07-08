import tensorflow as tf

kernel_initializer = 'he_normal'
kernel_regularizer = tf.keras.regularizers.l2(0.01)
bias_initializer = tf.keras.initializers.Constant(value=0.1)
leaky_relu_alpha = 0.02


def conv2d_block(x, filters):
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    return x


def create_model(
        input_shape=(64, 64, 3),
        num_layers=5,
        filters=64,
        num_classes=10,
        output_activation='softmax'):

    image_input = tf.keras.Input(shape=input_shape, name='img_input', dtype=tf.float32)
    x = tf.keras.layers.BatchNormalization()(image_input)

    for _ in range(num_layers):
        x = conv2d_block(x, filters=filters)
        filters *= 2

    x = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=(3, 3),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(x)

    logits = tf.keras.layers.GlobalAveragePooling2D(name='logits')(x)
    classes = tf.keras.layers.Activation(output_activation, name='classes')(logits)

    model = tf.keras.Model(inputs=[image_input], outputs=[logits, classes])
    return model
