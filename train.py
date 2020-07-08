import tensorflow as tf
import os
import json

from utils import tfrecords, visualize
from utils.augmentation import color, zoom, rotate, flip, gaussian, clip01

from CoffeeNet import create_model

# Load train/test data
train_ds = tfrecords.read(['./data/classification_train.tfrecord'], num_labels=7)
test_ds = tfrecords.read(['./data/classification_test.tfrecord'], num_labels=7)

# Apply augmentations
train_ds = zoom(train_ds, im_size=64)
train_ds = rotate(train_ds)
train_ds = flip(train_ds)
train_ds = clip01(train_ds)

# Set batchs
batch_size = 64
train_ds = train_ds.repeat().shuffle(buffer_size=10000).batch(batch_size)
test_ds = test_ds.repeat().shuffle(buffer_size=10000).batch(batch_size)

# Plot some images
visualize.plot_dataset(train_ds)

# Define model

model_name = 'CoffeeNet6'
model = create_model(
        input_shape=(64, 64, 3),
        num_layers=5,
        filters=64,
        num_classes=6,
        output_activation='softmax')

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    loss={'logits': tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.2)},
    metrics={'logits': [tf.keras.metrics.CategoricalAccuracy()]}
)
model.summary()

# Save model
savedir = os.path.join('results', model_name)
if not os.path.isdir(savedir):
    os.mkdir(savedir)

json_config = model.to_json()
with open(savedir + '/model.json', 'w') as f:
    json.dump(json_config, f)

# Save weights
model.save_weights(savedir + '/epoch-0000.h5')
filepath = savedir + '/epoch-{epoch:04d}.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, verbose=1, period=10)

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
    train_ds,
    steps_per_epoch=400,
    epochs=60,
    verbose=1,
    validation_data=test_ds,
    validation_freq=1,
    validation_steps=40,
    callbacks=[checkpoint, tb_callback]
)
