import tensorflow as tf

import matplotlib.pyplot as plt

from utils.labelmap import label_names


def plot_dataset(dataset):
    for data in dataset:
        imgs, labels = data
        print(imgs.shape)
        print(labels.shape)

        rows = 8
        columns = 8
        fig = plt.figure(figsize=(8, 8))
        for i in range(0, columns * rows):
            ax = fig.add_subplot(rows, columns, i + 1)
            ax.text(0, -3, label_names[labels[i]], fontsize=8)
            ax.imshow(imgs[i])
            ax.axis('off')

        plt.show()
        break


def normalize(x, y):
    x = tf.div(x, 255.)
    return x, y


def clip01(x, y):
    x = tf.clip_by_value(x, 0, 1)
    return x, y
