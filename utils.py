import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from labelmap import label_names


def plot_dataset(dataset):
    for data in dataset:
        imgs, labels = data

        rows = 8
        columns = 8
        fig = plt.figure(figsize=(8, 8))
        for i in range(0, columns * rows):
            img = imgs[i]
            label = labels[i]

            pred = np.argmax(label)
            name = label_names[pred]

            ax = fig.add_subplot(rows, columns, i + 1)
            ax.text(0, -3, name, fontsize=8)
            ax.imshow(img)
            ax.axis('off')

        plt.show()
        break


def normalize(x, y):
    x = tf.divide(x, 255.)
    return x, y


def clip01(x, y):
    x = tf.clip_by_value(x, 0, 1)
    return x, y
