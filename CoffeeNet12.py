import tensorflow as tf

from utils import labelmap
from utils import model as cnn

model_id = 'CoffeeNet12'


def model(x, is_training):
    with tf.name_scope('INPUT'):
        x = tf.truediv(tf.cast(x, tf.float32), 255.0)
        print("INPUT " + str(x.shape))

    x = cnn.conv2d(x, w=64, k=3, s=1)
    x = cnn.conv2d(x, w=64, k=3, s=1)
    x = cnn.maxpool(x, k=3, s=2)

    x = cnn.conv2d(x, w=128, k=3, s=1)
    x = cnn.conv2d(x, w=128, k=3, s=1)
    x = cnn.maxpool(x, k=3, s=2)

    x = cnn.conv2d(x, w=256, k=3, s=1)
    x = cnn.conv2d(x, w=256, k=3, s=1)
    x = cnn.maxpool(x, k=3, s=2)

    x = cnn.conv2d(x, w=512, k=3, s=1)
    x = cnn.conv2d(x, w=512, k=3, s=1)
    x = cnn.conv2d(x, w=512, k=3, s=1)
    x = cnn.maxpool(x, k=3, s=2)

    x = tf.layers.flatten(x)

    x = cnn.dense(x, w=1024)
    x = tf.layers.dropout(inputs=x, rate=0.50, training=is_training)

    x = cnn.dense(x, w=1024)
    x = tf.layers.dropout(inputs=x, rate=0.50, training=is_training)

    x = cnn.dense(x, w=labelmap.count, activation=None)

    return x
