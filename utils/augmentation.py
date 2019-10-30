import tensorflow as tf
import numpy as np

from utils import utils


def apply(dataset, im_size=64):
    def rotate(x, y):
        coin = tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        x = tf.image.rot90(x, coin)
        return x, y

    def flip(x, y):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        return x, y

    def color(x, y):
        x = tf.image.random_hue(x, 0.02)
        x = tf.image.random_saturation(x, 0.95, 1.05)
        x = tf.image.random_brightness(x, 0.02)
        x = tf.image.random_contrast(x, 0.95, 1.05)
        return x, y

    def zoom(x, y):
        scales = list(np.arange(0.9, 1.0, 0.01))
        boxes = np.zeros((len(scales), 4))

        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]

        def random_crop(img):
            crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(im_size, im_size))
            return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

        choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        x = tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))
        return x, y

    augmentations = [flip, color, zoom, rotate]
    for f in augmentations:
        dataset = dataset.map(f, num_parallel_calls=4)

    dataset = dataset.map(utils.clip01, num_parallel_calls=4)
    return dataset
