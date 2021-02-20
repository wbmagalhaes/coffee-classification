import tensorflow as tf

import os
import json

from utils import tfrecords, visualize
from utils.augmentation import color, zoom, rotate, flip, gaussian, clip01
from utils.labelmap import label_names

import math


def load_datasets(train_filenames, valid_filenames, batch_size):
    # Load train/test data
    train_ds = tfrecords.read_tfrecord(filenames=train_filenames)
    valid_ds = tfrecords.read_tfrecord(filenames=valid_filenames)

    # Apply augmentations
    train_ds = zoom(train_ds)
    train_ds = rotate(train_ds)
    train_ds = flip(train_ds)

    # train_ds = color(train_ds)
    # train_ds = gaussian(train_ds)

    train_ds = clip01(train_ds)

    train_steps = steps(train_ds, batch_size)
    valid_steps = steps(valid_ds, batch_size)

    # Set batchs
    train_ds = train_ds.repeat().shuffle(buffer_size=10000).batch(batch_size)
    valid_ds = valid_ds.repeat().shuffle(buffer_size=10000).batch(batch_size)

    return train_ds, valid_ds, train_steps, valid_steps


def create_model(
        model_name,
        input_shape=(64, 64, 3),
        num_layers=5,
        filters=64,
        kernel_initializer='he_normal',
        l2=0.01,
        bias_value=0.1,
        leaky_relu_alpha=0.02,
        output_activation='softmax',
        lr=1e-4,
        label_smoothing=0.2):

    # Define model
    image_input = tf.keras.Input(shape=input_shape, name='img_input', dtype=tf.float32)
    x = tf.keras.layers.BatchNormalization()(image_input)

    for _ in range(num_layers):
        x = conv2d_block(x, filters, kernel_initializer, tf.keras.regularizers.l2(l2), tf.keras.initializers.Constant(value=bias_value), leaky_relu_alpha)
        filters *= 2

    x = tf.keras.layers.Conv2D(
        filters=len(label_names),
        kernel_size=(3, 3),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        bias_initializer=tf.keras.initializers.Constant(value=bias_value),
        padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(x)

    logits = tf.keras.layers.GlobalAveragePooling2D(name='logits')(x)
    classes = tf.keras.layers.Activation(output_activation, name='classes')(logits)

    model = tf.keras.Model(inputs=[image_input], outputs=[logits, classes])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss={'logits': tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing)},
        metrics={'logits': [tf.keras.metrics.CategoricalAccuracy()]}
    )

    model.summary()

    # Save model
    if not os.path.isdir('models'):
        os.mkdir('models')

    savedir = os.path.join('models', model_name)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    json_config = model.to_json()
    with open(savedir + '/model.json', 'w') as f:
        json.dump(json_config, f)

    # Save weights
    model.save_weights(savedir + '/epoch-0000.h5')
    filepath = savedir + '/epoch-{epoch:04d}.h5'
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, verbose=1, period=10)

    # Initialize Tensorboard visualization
    logdir = os.path.join('logs', model_name)
    tensorboar_cb = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
        write_graph=True,
        profile_batch=0,
        update_freq='epoch'
    )

    return model, [checkpoint_cb, tensorboar_cb]


def steps(ds, batch_size):
    n = 0
    for data in ds:
        n += 1

    return math.ceil(n / batch_size)


def train(
        model,
        train_ds,
        valid_ds,
        train_steps,
        valid_steps,
        epochs=60,
        batch_size=64,
        callbacks=None):

    history = model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        epochs=epochs,
        verbose=1,
        validation_data=valid_ds,
        validation_freq=1,
        validation_steps=valid_steps,
        callbacks=callbacks
    )

    return history


def conv2d_block(x, filters, kernel_initializer, kernel_regularizer, bias_initializer, leaky_relu_alpha):
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
