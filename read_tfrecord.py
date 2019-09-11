import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import config
from utils import labelmap
from utils.tfrecords import get_data
from collections import defaultdict
import cv2 as cv

imgs, labels, _ = get_data(filenames=[config.VALIDATION_PATH], shuffle=True)
print(len(imgs))

ims = tf.placeholder(tf.float32, [None, config.IMG_SIZE, config.IMG_SIZE, 3])

label_counter = defaultdict(int)
for label in labels:
    label_counter[np.argmax(label)] += 1

print('============')
for l in label_counter:
    label = labelmap.name_of_idx(l)
    print(label, ':', label_counter[l])
print('============')

for x, l in zip(imgs, labels):
    label = labelmap.name_of_idx(np.argmax(l))

    x *= 255.0
    x = x.astype(np.uint8)
    x = cv.cvtColor(x, cv.COLOR_Lab2RGB)

    plt.imshow(x)
    plt.title(label)
    plt.show()
