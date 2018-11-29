import tensorflow as tf
import numpy as np
import sys
import os
import cv2

from utils import labelmap
from utils import config

# int64 is used for numeric values


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# float is used for numeric values


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# bytes is used for string/char values


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def write_tfrecords(filepath, imgs_data):
    # Initiating the writer and creating the tfrecords file.
    writer = tf.python_io.TFRecordWriter(filepath)

    for img_data in imgs_data:
        # Read the image data
        img = img_data['image']
        label = img_data['label']

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': bytes_feature(tf.compat.as_bytes(img.tostring())),
            'label': int64_feature(label)
        }))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def get_dataset(filenames, batch_size=config.BATCH_SIZE, shuffle=True):
    print('Config dataset.')
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(serialized_example):
        features = tf.parse_single_example(serialized_example, features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })

        image = tf.decode_raw(features['image'], tf.uint8)
        label = tf.cast(features['label'], tf.int64)

        image = tf.reshape(
            image, [config.IMG_SIZE, config.IMG_SIZE, 3], name="image")
        label = tf.one_hot(label, labelmap.count, name="label")

        return image, label

    dataset = dataset.map(parser)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(batch_size)

    return dataset


def get_data(filenames, shuffle):
    dataset = get_dataset(filenames, batch_size=10000, shuffle=shuffle)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    print('Reading dataset.')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)

        images, labels = sess.run(next_element)

    print('End of dataset.')
    return images, labels
