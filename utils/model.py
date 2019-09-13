import tensorflow as tf

from utils import labelmap

n_layer = 0

kernel_initializer = tf.initializers.he_uniform()
kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001)
bias_initializer = tf.initializers.zeros()
bias_regularizer = None


def conv2d(x, w, k, s, activation=tf.nn.leaky_relu):
    global n_layer
    n_layer += 1
    name = 'CONV' + str(n_layer)

    out = tf.layers.conv2d(
        inputs=x,
        filters=w,
        kernel_size=k,
        strides=s,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        padding='SAME',
        name=name)

    print(name, out.shape)
    return out


def conv2d_t(x, w, k, s, activation=tf.nn.leaky_relu):
    global n_layer
    n_layer += 1
    name = 'CONVT' + str(n_layer)

    out = tf.layers.conv2d_transpose(
        inputs=x,
        filters=w,
        kernel_size=k,
        strides=s,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        padding='SAME',
        name=name)

    print(name, out.shape)
    return out


def maxpool(x, k, s):
    global n_layer
    name = 'POOL' + str(n_layer)

    out = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=k,
        strides=s,
        padding='SAME',
        name=name)

    print(name, out.shape)
    return out


def gap(x, w, k, s, p, activation=tf.nn.leaky_relu):
    global n_layer
    n_layer += 1
    name = 'GAP' + str(n_layer)

    x = tf.layers.conv2d(
        inputs=x,
        filters=w,
        kernel_size=k,
        strides=s,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        padding='SAME',
        name=name)

    x = tf.layers.average_pooling2d(
        inputs=x,
        pool_size=p,
        strides=1,
        padding='SAME')

    out = tf.reduce_mean(x, axis=[1, 2])
    print(name, out.shape)
    return out


def dense(x, w, activation=tf.nn.leaky_relu):
    global n_layer
    n_layer += 1
    name = 'DENSE' + str(n_layer)

    out = tf.layers.dense(
        inputs=x,
        units=w,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        name=name)

    print(name, out.shape)
    return out


def loss_function(y_pred, y_true, weights=1, label_smoothing=0.2):
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=y_pred, label_smoothing=label_smoothing, weights=weights)
    l2_loss = tf.losses.get_regularization_loss()

    return tf.reduce_mean(cross_entropy) + l2_loss


def accuracy_function(y_pred, y_true):
    return tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
