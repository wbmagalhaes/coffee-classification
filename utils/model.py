import tensorflow as tf

from utils import labelmap

n_layer = 0


def conv2d(x, w, k, s, activation=tf.nn.relu):
    global n_layer
    n_layer += 1
    name = 'CONV' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.conv2d(
            inputs=x, filters=w, kernel_size=k, strides=s, activation=activation, padding='SAME', name=name)

    print(name, out.shape)
    return out


def conv2d_t(x, w, k, s, activation=tf.nn.relu):
    global n_layer
    n_layer += 1
    name = 'CONVT' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.conv2d_transpose(
            inputs=x, filters=w, kernel_size=k, strides=s, activation=activation, padding='SAME', name=name)

    print(name, out.shape)
    return out


def maxpool(x, k, s):
    global n_layer
    name = 'POOL' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.max_pooling2d(
            inputs=x, pool_size=k, strides=s, padding='SAME', name=name)

    print(name, out.shape)
    return out


def dense(x, w, activation=tf.nn.relu):
    global n_layer
    n_layer += 1
    name = 'DENSE' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.dense(
            inputs=x, units=w, activation=activation, name=name)

    print(name, out.shape)
    return out


def loss_function(y_pred, y_true):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true, logits=y_pred)

    return tf.reduce_mean(cross_entropy)


def accuracy_function(y_pred, y_true):
    return tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
