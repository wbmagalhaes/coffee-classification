import os
import tensorflow as tf

from random import shuffle
from utils import data_reader

from utils.labelmap import label_names
from utils import visualize
from utils.augmentation import color, zoom, rotate, flip, gaussian, clip01


def load_datafiles(input_dir, im_size=64, random=True, training_percentage=0.8, n_files=(1, 1, 1)):
    dataset = data_reader.load(input_dir, cut_size=im_size, bg_color=(0, 0, 0))

    if random:
        shuffle(dataset)

    train_num = int(len(dataset) * training_percentage)
    teste_num = int(len(dataset) * (1 - training_percentage)) // 2

    train_dataset = dataset[:train_num]
    valid_dataset = dataset[train_num:train_num + teste_num]
    teste_dataset = dataset[train_num + teste_num:]

    return train_dataset, valid_dataset, teste_dataset


def save_tfrecords(data, name, output_dir, n=1):
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
        generator, output_types=tf.string, output_shapes=())
    writer.write(dataset)


def read_tfrecord(filenames, img_size=64):
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
        image = tf.reshape(image, [img_size, img_size, 3])

        label = tf.one_hot(label, len(label_names))

        return image, label

    dataset = raw_dataset.map(parser, num_parallel_calls=4)
    return dataset


def show_dataset(path, batch=36, augment=True):
    ds = read_tfrecord([path])

    visualize.plot_dataset(ds.batch(batch))

    if augment:
        ds = rotate(ds)
        ds = flip(ds)
        ds = clip01(ds)

        visualize.plot_dataset(ds.batch(batch))


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
