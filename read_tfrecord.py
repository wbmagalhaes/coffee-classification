import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import config
from utils import labelmap
from utils.tfrecords import get_data

imgs, labels = get_data(filenames=[config.TESTING_PATH], shuffle=True)

print(len(imgs))
for i in range(len(imgs)):
    plt.imshow(imgs[i])
    label_id = np.argmax(labels[i])
    label = labelmap.labels[label_id]
    plt.title(label)
    plt.show()
