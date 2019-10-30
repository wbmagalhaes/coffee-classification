
import itertools
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

from CoffeeNet6 import create_model

from utils.labelmap import label_names

x_test, y_test = [], []

model = create_model()
model.load_weights('./results/coffeenet6.h5')

y_pred = model.predict(x_test/255.)


def plot_predictions(images, predictions):
    n = images.shape[0]
    nc = int(np.ceil(n / 4))
    _, axes = plt.subplots(nc, 4)
    for i in range(nc * 4):
        y = i // 4
        x = i % 4
        axes[x, y].axis('off')

        label = label_names[np.argmax(predictions[i])]
        confidence = np.max(predictions[i]) * 100
        if i > n:
            continue

        axes[x, y].imshow(images[i] / 255.)
        axes[x, y].text(0, -3, f'{label} {confidence:.1f}', fontsize=12)

    plt.gcf().set_size_inches(8, 8)


plot_predictions(x_test[:16], y_pred[:16])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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


# Generate the confusion matrix
pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, pred)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=label_names)

plt.show()
