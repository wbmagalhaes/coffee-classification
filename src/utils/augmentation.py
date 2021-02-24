import tensorflow as tf
import numpy as np


def color(
        dataset,
        hue=0.05,
        saturation=(0.9, 1.05),
        brightness=0.1,
        contrast=(0.9, 1.05)):

    def apply(x, y):
        x = tf.image.random_hue(x, hue)
        x = tf.image.random_saturation(x, saturation[0], saturation[1])
        x = tf.image.random_brightness(x, brightness)
        x = tf.image.random_contrast(x, contrast[0], contrast[1])
        return x, y

    return dataset.map(apply, num_parallel_calls=4)


def zoom(dataset, im_size=64):
    def apply(x, y):
        scales = list(np.arange(0.8, 1.0, 0.05))
        boxes = np.zeros((len(scales), 4))

        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]

        def random_crop(img):
            crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(
                len(scales)), crop_size=(im_size, im_size))
            return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

        choice = tf.random.uniform(
            shape=[], minval=0., maxval=1., dtype=tf.float32)
        x = tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))
        return x, y

    return dataset.map(apply, num_parallel_calls=4)


def rotate(dataset):
    def apply(x, y):
        coin = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        x = tf.image.rot90(x, coin)
        return x, y

    return dataset.map(apply, num_parallel_calls=4)


def flip(dataset):
    def apply(x, y):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        return x, y

    return dataset.map(apply, num_parallel_calls=4)


def gaussian(dataset, stddev=1/255):
    def apply(x, y):
        noise = tf.random.normal(shape=tf.shape(
            x), mean=0.0, stddev=stddev, dtype=tf.float32)
        x = x + noise
        return x, y

    return dataset.map(apply, num_parallel_calls=4)


def clip01(dataset):
    def apply(x, y):
        x = tf.clip_by_value(x, 0, 1)
        return x, y

    return dataset.map(apply, num_parallel_calls=4)
