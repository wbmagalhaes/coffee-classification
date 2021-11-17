
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from coffee_classification.utils.neural_net import load_datasets, prepare_datasets
from coffee_classification.utils.labelmap import label_names


train_filenames = ['data/train_dataset.tfrecord']
valid_filenames = ['data/valid_dataset.tfrecord']

train_ds, valid_ds = load_datasets(train_filenames, valid_filenames)
train_ds, valid_ds, train_steps, valid_steps = prepare_datasets(
    train_ds,
    valid_ds,
    repeat=True,
    shuffle=True,
    batch_size=64
)

print(train_steps, valid_steps)

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)

#         img = (images[i].numpy() * 255).astype("uint8")
#         plt.imshow(img)

#         label = label_names[np.argmax(labels[i].numpy())]
#         plt.title(label)
#         plt.axis("off")

# plt.show()

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(1),
])

# for image, _ in train_ds.take(1):
#     plt.figure(figsize=(10, 10))
#     first_image = image[0]
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
#         img = (augmented_image[0].numpy() * 255).astype("uint8")
#         plt.imshow(img)
#         plt.axis('off')

# plt.show()

base_model = tf.keras.applications.resnet_v2.ResNet152V2(
    input_shape=(64, 64, 3),
    include_top=False,
    weights='imagenet'
)

image_batch, label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)

base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

prediction_layer = tf.keras.layers.Dense(len(label_names))
prediction_batch = prediction_layer(feature_batch_average)

inputs = tf.keras.Input(shape=(64, 64, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)

loss0, accuracy0 = model.evaluate(
    valid_ds,
    steps=valid_steps,
)

print(f'initial loss: {loss0:.2f}')
print(f'initial accuracy: {accuracy0:.2f}')

# history = model.fit(
#     train_ds,
#     steps_per_epoch=train_steps,
#     epochs=10,
#     validation_data=valid_ds,
#     validation_freq=1,
#     validation_steps=valid_steps,
# )

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()), 1])
# plt.title('Training and Validation Accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0, 1.0])
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')

# plt.show()
