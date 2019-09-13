import tensorflow as tf

from utils import labelmap
from utils import model as cnn

model_id = 'CoffeeNet6'


def model(x):
    print("INPUT " + str(x.shape))
    x = tf.image.per_image_standardization(x)

    x = cnn.conv2d(x, w=64, k=5, s=1)
    x = cnn.maxpool(x, k=3, s=2)

    x = cnn.conv2d(x, w=128, k=5, s=1)
    x = cnn.maxpool(x, k=3, s=2)

    x = cnn.conv2d(x, w=256, k=3, s=1)
    x = cnn.maxpool(x, k=3, s=2)

    x = cnn.conv2d(x, w=512, k=3, s=1)
    x = cnn.maxpool(x, k=3, s=2)

    x = cnn.conv2d(x, w=1024, k=3, s=1)
    x = cnn.maxpool(x, k=3, s=2)

    x = cnn.gap(x, w=labelmap.count, k=3, s=1, p=4, activation=None)

    return x
