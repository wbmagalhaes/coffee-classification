import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

from utils import tfrecords, augmentation, utils
from CoffeeNet6 import create_model

tf.enable_eager_execution()

# Load dataset
dataset = tfrecords.read_tfrecord(['./data/coffee_data.tfrecord'])

dataset = dataset.map(utils.normalize, num_parallel_calls=4)
dataset = augmentation.apply(dataset)

dataset = dataset.repeat().shuffle(buffer_size=10000).batch(64)

# Plot some images
utils.plot_dataset(dataset)

# Define model
model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'])

model.summary()

# Tensorboard visualization
logdir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=10,
    write_graph=True)

# Training
model.fit(
    dataset,
    steps_per_epoch=400,
    epochs=5,
    verbose=1,
    callbacks=[tb_callback]
)

# Save weights
model.save_weights('./results/coffeenet6.h5', overwrite=True)
