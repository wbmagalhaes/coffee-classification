import tensorflow as tf

import os

from utils import tfrecords, augmentation, other, visualize
from CoffeeNet6 import create_model

model_name = 'CoffeeNet6'

# Load train data
train_dataset = tfrecords.read(['./data/data_train.tfrecord'])
train_dataset = train_dataset.map(other.normalize, num_parallel_calls=4)

# Load test data
test_dataset = tfrecords.read(['./data/data_test.tfrecord'])
test_dataset = test_dataset.map(other.normalize, num_parallel_calls=4)

# Apply augmentations
train_dataset = augmentation.apply(train_dataset)

# Set batchs
train_dataset = train_dataset.repeat().shuffle(buffer_size=10000).batch(64)
test_dataset = test_dataset.repeat().shuffle(buffer_size=10000).batch(64)

# Plot some images
# visualize.plot_dataset(train_dataset)

# Define model
model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    loss={
        'logits': tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.2),
    },
    metrics={
        'logits': tf.keras.metrics.CategoricalAccuracy(name='acc')
    }
)
# model.summary()

# Tensorboard visualization
logdir = os.path.join('logs', model_name)
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=1,
    write_graph=True,
    profile_batch=0,
    update_freq='epoch'
)

# Training
history = model.fit(
    train_dataset,
    steps_per_epoch=400,
    epochs=60,
    verbose=1,
    validation_data=test_dataset,
    validation_freq=1,
    validation_steps=40,
    callbacks=[tb_callback]
)

# Save weights
model.save_weights('./results/coffeenet6.h5', overwrite=True)
