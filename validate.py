
from sklearn.metrics import classification_report, confusion_matrix
import itertools

import matplotlib.pyplot as plt
import numpy as np

from CoffeeNet6 import create_model

import data_reader
from labelmap import label_names

validation_paths = [
    'C:/Users/Usuario/Desktop/cafe_imgs/cut_samples/84A',
    'C:/Users/Usuario/Desktop/cafe_imgs/cut_samples/248A'
]
data = data_reader.load(validation_paths)

x_test, y_test = zip(*data)
x_test = np.array(x_test).astype(np.float32) / 255.
y_test = np.array(y_test).astype(np.float32)

model = create_model()
model.load_weights('./results/coffeenet6.h5')

_, classes = model.predict(x_test)


def plot_predictions(images, predictions):
    n = images.shape[0]
    nc = int(np.ceil(n / 4))
    _, axes = plt.subplots(nc, 4)
    for i in range(nc * 4):
        y = i // 4
        x = i % 4
        axes[x, y].axis('off')

        pred = predictions[i]
        pred = np.argmax(pred)

        name = label_names[pred]
        confidence = np.max(predictions[i]) * 100
        if i > n:
            continue

        axes[x, y].imshow(images[i])
        axes[x, y].text(0, -3, f'{name} {confidence:.1f}', fontsize=12)

    plt.gcf().set_size_inches(8, 8)
    plt.show()


plot_predictions(x_test[:16], classes[:16])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='Blues'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

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


# Generate the confusion matrix
classes = np.argmax(classes, axis=1)

classes = np.append(classes, [3])
y_test = np.append(y_test, [3])

cm = confusion_matrix(y_test, classes)

# Plot confusion matrix
plot_confusion_matrix(cm, classes=label_names)
