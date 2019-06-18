import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from utils import config
from utils.tfrecords import get_data


def plot_colors(hist, centroids, size=(300, 50)):
    w, h = size
    bar = np.zeros((h, w, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * w)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), h), color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def get_dominant_color(image, k=4):
    """
    create a histogram with k clusters
    :param: image, k
    :return: hist, clt
    """

    img = image.copy()

    img = img.reshape((img.shape[0] * img.shape[1], 3))  # represent as (row * column, channel)
    clt = KMeans(n_clusters=k)  # cluster number
    clt.fit(img)

    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    colors = clt.cluster_centers_

    return (list(t) for t in zip(*sorted(zip(hist, colors), reverse=True)))


data_x, data_y = get_data([config.VALIDATION_PATH, config.TESTING_PATH], shuffle=False)

for img in data_x:
    hist, colors = get_dominant_color(img, 5)
    bar = plot_colors(hist, colors, (img.shape[0], 16))

    result = np.concatenate((img, bar), axis=0)

    plt.axis('off')
    plt.imshow(result)
    plt.show()
