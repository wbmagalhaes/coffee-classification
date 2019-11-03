import math

import numpy as np
import matplotlib.pyplot as plt

from utils.labelmap import label_names, defect_values

from sklearn.metrics import classification_report, confusion_matrix
import itertools


def plot_dataset(dataset, figsize=8, fontsize=10):
    for data in dataset:
        imgs, labels = data

        n = len(imgs)
        rows = int(math.ceil(math.sqrt(n)))

        fig = plt.figure(figsize=(figsize, figsize))
        for i in range(rows * rows):
            if i >= n:
                break

            img = imgs[i]
            label = labels[i]

            pred = np.argmax(label)
            name = label_names[pred]

            ax = fig.add_subplot(rows, rows, i + 1)
            ax.text(0, -3, name, fontsize=fontsize)
            ax.imshow(img)
            ax.axis('off')

        plt.show()
        break


def plot_images(x_data, y_true, y_pred=None, figsize=8, fontsize=10):
    def get_label(y):
        try:
            len(y)
            y = np.argmax(y)
        finally:
            return label_names[int(y)][:3]

    n = len(x_data)

    rows = int(math.ceil(math.sqrt(n)))

    fig = plt.figure(figsize=(figsize, figsize))
    for i in range(rows * rows):
        if i >= n:
            break

        img = x_data[i]
        ax = fig.add_subplot(rows, rows, i + 1)

        if y_pred is not None:
            conf = np.max(y_pred[i]) * 100
            pred = get_label(y_pred[i])

        true = get_label(y_true[i])

        if y_pred is None:
            ax.text(0, -3, f'{true}', fontsize=fontsize, color='black')
        elif true == pred:
            ax.text(0, -3, f'{pred} {conf:.1f}%', fontsize=fontsize, color='black')
        else:
            ax.text(0, -3, f'{true} - {pred} {conf:.1f}%', fontsize=fontsize, color='red')

        ax.imshow(img)
        ax.axis('off')

    plt.show()


def get_label_list(ys):
    try:
        len(ys[0])
        ys = np.argmax(ys, axis=1)
    finally:
        return ys


def plot_confusion_matrix(y_true, y_pred, normalize=False, cmap='Blues'):
    y_true = get_label_list(y_true)
    y_pred = get_label_list(y_pred)

    report = classification_report(y_true, y_pred)
    print(report)

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def count_defects(ys):
    ys = get_label_list(ys)

    defects = {}
    for label in label_names:
        defects[label] = 0

    for y in ys:
        defects[label_names[int(y)]] += 1

    return defects


def sum_defects(defects):
    total = 0

    for label in label_names:
        total += defects[label] * defect_values[label]

    return total
