import tensorflow as tf


def aug_data(images):
    with tf.name_scope('augument'):
        batch_size = tf.shape(images)[0]

        with tf.name_scope('horizontal_flip'):
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            images = tf.where(coin, tf.image.flip_left_right(images), images)

        with tf.name_scope('vertical_flip'):
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            images = tf.where(coin, tf.image.flip_up_down(images), images)

        with tf.name_scope('rotate'):
            angle = tf.cast(tf.random_uniform([], 0, 4.0), tf.int32)
            images = tf.image.rot90(images, angle)

    return images
