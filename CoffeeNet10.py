import tensorflow as tf

from utils import labelmap
from utils import model as cnn

model_id = 'CoffeeNet10'


def model(x):
    print("INPUT " + str(x.shape))

    x = cnn.conv2d(x, w=32, k=5, s=1)
    x = cnn.conv2d(x, w=32, k=5, s=1)
    x = cnn.maxpool(x, k=3, s=2)

    x = cnn.conv2d(x, w=64, k=5, s=1)
    x = cnn.conv2d(x, w=64, k=5, s=1)
    x = cnn.maxpool(x, k=3, s=2)

    x = cnn.conv2d(x, w=128, k=3, s=1)
    x = cnn.conv2d(x, w=128, k=3, s=1)
    x = cnn.maxpool(x, k=3, s=2)

    x = cnn.conv2d(x, w=256, k=3, s=1)
    x = cnn.conv2d(x, w=256, k=3, s=1)
    x = cnn.maxpool(x, k=3, s=2)

    x = tf.layers.flatten(x)

    x = cnn.dense(x, w=512)

    x = cnn.dense(x, w=labelmap.count, activation=None)
    return x
