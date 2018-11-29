import tensorflow as tf

from utils import labelmap

model_id = 'simple_3'

initializer = tf.variance_scaling_initializer()

n_layer = 0


def conv2d(x, w, k, s, activation=tf.nn.relu):
    global n_layer
    n_layer += 1
    name = 'CONV' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.conv2d(inputs=x, filters=w, kernel_size=k, strides=s, activation=activation,
                               kernel_initializer=initializer, bias_initializer=initializer, padding='SAME')

    print(name, out.shape)
    return out


def maxpool(x, s, p):
    global n_layer
    name = 'POOL' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.max_pooling2d(
            inputs=x, pool_size=p, strides=s, padding='SAME')

    print(name, out.shape)
    return out


def dense(x, w, activation=tf.nn.relu):
    global n_layer
    n_layer += 1
    name = 'DENSE' + str(n_layer)

    with tf.name_scope(name):
        out = tf.layers.dense(inputs=x, units=w, activation=activation,
                              kernel_initializer=initializer, bias_initializer=initializer)

    print(name, out.shape)
    return out


def model(x, is_training):
    with tf.name_scope('INPUT'):
        x = tf.truediv(tf.cast(x, tf.float32), 255.0)
        print("INPUT " + str(x.shape))

    x = conv2d(x, w=64, k=5, s=1)
    x = maxpool(x, s=2, p=2)

    x = conv2d(x, w=128, k=5, s=1)
    x = maxpool(x, s=2, p=2)

    x = conv2d(x, w=256, k=3, s=1)
    x = maxpool(x, s=2, p=2)

    x = tf.layers.flatten(x)

    x = dense(x, w=512)
    x = tf.layers.dropout(inputs=x, rate=0.25, training=is_training)

    x = dense(x, w=labelmap.count, activation=None)

    return x


def loss_function(y_pred, y_true):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true, logits=y_pred)
    return tf.reduce_mean(cross_entropy)


def accuracy_function(y_pred, y_true):
    return tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
