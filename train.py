import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

from utils import tfrecords, augmentation, utils
from CoffeeNet6 import create_model

tf.enable_eager_execution()

# Load train data
train_dataset = tfrecords.read_tfrecord(['./data/data_train.tfrecord'])
train_dataset = train_dataset.map(utils.normalize, num_parallel_calls=4)

# Load test data
test_dataset = tfrecords.read_tfrecord(['./data/data_test.tfrecord'])
test_dataset = test_dataset.map(utils.normalize, num_parallel_calls=4)

# Apply augmentations
train_dataset = augmentation.apply(train_dataset)

# Plot some images
# utils.plot_dataset(train_dataset)

# Set batchs
train_dataset = train_dataset.repeat().shuffle(buffer_size=10000).batch(64)
test_dataset = test_dataset.repeat().shuffle(buffer_size=10000).batch(64)

# Define model
model = create_model()

opt = tf.keras.optimizers.Adam(lr=1e-4),
loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    label_smoothing=0.2)

model.compile(
    optimizer=opt,
    loss=loss,
    metrics=['categorical_accuracy'])

model.summary()

# Tensorboard visualization
logdir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=10,
    write_graph=True)

# Training
model.fit(
    train_dataset,
    steps_per_epoch=400,
    epochs=100,
    verbose=1,
    validation_data=test_dataset,
    validation_steps=32,
    callbacks=[tb_callback]
)

# Save weights
model.save_weights('./results/coffeenet6.h5', overwrite=True)
