import os
import tensorflow as tf

from random import shuffle
from coffee_classification.utils import data_reader

from coffee_classification.utils.labelmap import label_names
from coffee_classification.utils import visualize
from coffee_classification.utils.augmentation import color, zoom, rotate, flip, gaussian, clip01


def save_tfrecord(data, name, output_dir, n=1):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    path = os.path.join(output_dir, name)

    if n == 1:
        write_tfrecord(f"{path}.tfrecord", data)

    else:

        size = len(data) // n
        for i in range(0, n + 1):
            splitted = data[size * i:size * (i + 1)]
            if len(splitted) > 0:
                write_tfrecord(f"{path}{i}.tfrecord", splitted)


def write_tfrecord(filename, data):
    writer = tf.data.experimental.TFRecordWriter(filename)

    def serialize_example(image, label):
        image = tf.compat.as_bytes(image.tobytes())

        feature = {
            'image': bytes_feature(image),
            'label': int64_feature(label)
        }

        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def generator():
        for features in data:
            yield serialize_example(*features)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=tf.string,
        output_shapes=()
    )

    writer.write(dataset)


def read_tfrecord(filenames, im_size=64):
    raw_dataset = tf.data.TFRecordDataset(filenames)

    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    def parser(example_proto):
        features = tf.io.parse_single_example(
            example_proto, feature_description)

        raw_image = tf.io.decode_raw(features['image'], tf.uint8)
        label = tf.cast(features['label'], tf.int64)

        image = tf.cast(raw_image, tf.float32) / 255.
        image = tf.reshape(image, [im_size, im_size, 3])

        label = tf.one_hot(label, len(label_names))

        return image, label

    return raw_dataset.map(parser, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)


def show_dataset(dataset, batch=36, augment=True):
    visualize.plot_dataset(dataset.batch(batch))

    if augment:
        dataset = rotate(dataset)
        dataset = flip(dataset)
        dataset = clip01(dataset)

        visualize.plot_dataset(dataset.batch(batch))


def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
