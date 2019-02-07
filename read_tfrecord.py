import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import config
from utils import labelmap
from utils.tfrecords import get_data
from collections import defaultdict

imgs, labels = get_data(filenames=[config.TRAINING_PATH], shuffle=True)

print(len(imgs))

label_counter = defaultdict(int)
for label in labels:
    label_counter[np.argmax(label)] += 1

print('============')
for l in label_counter:
    print(labelmap.name_of_idx(l),':', label_counter[l])
print('============')


for img, label in zip(imgs, labels):
    plt.imshow(img)
    label_id = np.argmax(label)
    label = labelmap.labels[label_id]
    plt.title(label)
    plt.show()
