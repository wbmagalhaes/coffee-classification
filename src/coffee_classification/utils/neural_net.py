import tensorflow as tf

import os
import json

from coffee_classification.utils import tfrecords
from coffee_classification.utils.augmentation import (
    color,
    zoom,
    rotate,
    flip,
    gaussian,
    clip01)
from coffee_classification.utils.labelmap import label_names

import math


def load_datasets(train_filenames, valid_filenames):
    # Load train/test data
    train_ds = tfrecords.read_tfrecord(filenames=train_filenames)
    valid_ds = tfrecords.read_tfrecord(filenames=valid_filenames)
    return train_ds, valid_ds


def apply_augmentations(dataset, aug=None):
    if aug is None:
        aug = ['zoom', 'rotate', 'flip', 'gaussian']

    if 'zoom' in aug:
        dataset = zoom(dataset)

    if 'rotate' in aug:
        dataset = rotate(dataset)

    if 'flip' in aug:
        dataset = flip(dataset)

    if 'color' in aug:
        dataset = color(dataset)

    if 'gaussian' in aug:
        dataset = gaussian(dataset)

    dataset = clip01(dataset)
    return dataset


def prepare_datasets(train_ds, valid_ds, repeat=True, shuffle=True, batch_size=64):
    train_steps = calculate_steps(train_ds, batch_size)
    valid_steps = calculate_steps(valid_ds, batch_size)

    if repeat:
        train_ds = train_ds.repeat()
        valid_ds = valid_ds.repeat()

    if shuffle:
        train_ds = train_ds.shuffle(buffer_size=10000)
        valid_ds = valid_ds.shuffle(buffer_size=10000)

    train_ds = train_ds.batch(batch_size)
    valid_ds = valid_ds.batch(batch_size)

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, valid_ds, train_steps, valid_steps


def create_model(
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
        x = conv2d_block(
            x,
            filters,
            kernel_initializer,
            tf.keras.regularizers.l2(l2),
            tf.keras.initializers.Constant(value=bias_value),
            leaky_relu_alpha)
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={'logits': tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing)},
        metrics={'logits': [tf.keras.metrics.CategoricalAccuracy()]}
    )

    model.summary()

    return model


def save_model(model, model_dir, log_dir=None):

    os.makedirs(model_dir, exist_ok=True)

    json_config = model.to_json()
    with open(model_dir + '/model.json', 'w') as f:
        json.dump(json_config, f)

    # Save weights
    model.save_weights(model_dir + '/epoch-0000.h5')
    filepath = model_dir + '/epoch-{epoch:04d}.h5'
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, verbose=1, period=10)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

        # Initialize Tensorboard visualization
        tensorboar_cb = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            profile_batch=0,
            update_freq='epoch'
        )

        return [checkpoint_cb, tensorboar_cb]

    return [checkpoint_cb]


def calculate_steps(ds, batch_size):
    n = len([0 for _ in ds])
    return math.ceil(n / batch_size)


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
