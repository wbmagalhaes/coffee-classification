import tensorflow as tf


def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tfrecord(filename, data):
    writer = tf.data.experimental.TFRecordWriter(filename)

    def serialize_example(image, label):
        image = tf.compat.as_bytes(image.tostring())

        feature = {
            'image': bytes_feature(image),
            'label': int64_feature(label)
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def generator():
        for features in data:
            yield serialize_example(*features)

    dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
    writer.write(dataset)


def read_tfrecord(filenames, img_size=64, num_labels=10):
    tf.enable_eager_execution()

    raw_dataset = tf.data.TFRecordDataset(filenames)

    feature_description = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }

    def parser(example_proto):
        features = tf.parse_single_example(example_proto, feature_description)

        raw_image = tf.decode_raw(features['image'], tf.uint8)
        label = tf.cast(features['label'], tf.int64)

        image = tf.cast(raw_image, tf.float32)
        image = tf.reshape(image, [img_size, img_size, 3])
        # label = tf.one_hot(label, num_labels)

        return image, label

    dataset = raw_dataset.map(parser, num_parallel_calls=4)
    return dataset
