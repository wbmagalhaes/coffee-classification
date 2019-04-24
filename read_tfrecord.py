import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import config
from utils import labelmap
from utils.tfrecords import get_data
from collections import defaultdict

imgs, labels = get_data(filenames=[config.VALIDATION_PATH], shuffle=True)

print(len(imgs))

label_counter = defaultdict(int)
for label in labels:
    label_counter[np.argmax(label)] += 1

print('============')
for l in label_counter:
    label = labelmap.name_of_idx(l)
    print(label, ':', label_counter[l])
print('============')

for img, l in zip(imgs, labels):
    label = labelmap.name_of_idx(np.argmax(l))

    plt.imshow(img)
    plt.title(label)
    plt.show()
