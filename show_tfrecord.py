import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from utils import tfrecords, augmentation, utils

tf.enable_eager_execution()

dataset = tfrecords.read_tfrecord(['./data/coffee_data.tfrecord']).shuffle(buffer_size=10000)
dataset = dataset.map(utils.normalize, num_parallel_calls=4)
utils.plot_dataset(dataset.batch(64))

dataset = augmentation.apply(dataset)
utils.plot_dataset(dataset.batch(64))
