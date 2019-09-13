import tensorflow as tf
import numpy as np
import time
import glob

from utils import config, labelmap
from utils.tfrecords import get_data, get_iterator
from utils.data_augment import aug_data
from utils.model import loss_function, accuracy_function

import CoffeeNet6 as CoffeeNet

training_dir = config.CHECKPOINT_DIR + CoffeeNet.model_id

print(f'Using model {CoffeeNet.model_id}')

with tf.name_scope('dataset_load'):
    train_x, train_y, train_w = get_data(filenames=[config.TRAINING_PATH], shuffle=True)
    test_x, test_y, test_w = get_data(filenames=[config.TESTING_PATH], shuffle=True)

    # train_iter, train_next_element = get_iterator([config.TRAINING_PATH], repeat=True)
    # test_iter, test_next_element = get_iterator([config.TESTING_PATH], batch_size=2000, repeat=True)

with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, config.IMG_SIZE, config.IMG_SIZE, 3], name='img_input')
    y = tf.placeholder(tf.float32, [None, labelmap.count], name='label_input')
    w = tf.placeholder(tf.float32, [None], name='loss_weights')

    augument_op = aug_data(x)

with tf.name_scope('neural_net'):
    model_result = CoffeeNet.model(x)

with tf.name_scope('result'):
    logits = tf.identity(model_result, name='logits')
    probs = tf.nn.softmax(logits, name='probs')
    label = tf.argmax(probs, 1, name='label')

with tf.name_scope('score'):
    y_true = tf.argmax(y, 1)
    loss_op = loss_function(y_pred=model_result, y_true=y, weights=w)
    accuracy_op = accuracy_function(y_pred=label, y_true=y_true)

tf.summary.scalar('score/loss', loss_op)
tf.summary.scalar('score/accuracy', accuracy_op)

global_step = tf.train.get_or_create_global_step()

learning_rate = tf.train.exponential_decay(
    learning_rate=config.LEARNING_RATE,
    global_step=global_step,
    decay_steps=config.DECAY_STEPS,
    decay_rate=config.DECAY_RATE,
    staircase=False
)

tf.summary.scalar('learning_rate', learning_rate)

step_per_sec = tf.placeholder(tf.float32, name='step_per_sec_op')
tf.summary.scalar('step_per_sec', step_per_sec)
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='AdamOpt')
    train_op = optimizer.minimize(loss_op, global_step=global_step, name='TrainOp')

merged_train = tf.identity(tf.summary.merge_all(), name='merged_train_op')

for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

merged_test = tf.identity(tf.summary.merge_all(), name='merged_test_op')
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(training_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(training_dir + '/test')

    print('Starting train...')

    time_i = time.time()

    # sess.run(train_iter.initializer)
    # sess.run(test_iter.initializer)

    for epoch in range(config.EPOCHS + 1):
        delta_time = time.time() - time_i
        time_i = time.time()

        if delta_time <= 0:
            delta_time = 1
        s_per_sec = 1.0 / delta_time

        lower_bound = (epoch * config.BATCH_SIZE) % len(train_x)
        upper_bound = lower_bound + config.BATCH_SIZE
        images = train_x[lower_bound:upper_bound]
        labels = train_y[lower_bound:upper_bound]
        weight = train_w[lower_bound:upper_bound]

        # images, labels, weight = sess.run(train_next_element)
        images = sess.run(augument_op, feed_dict={x: images})

        feed_dict = {x: images, y: labels, w: weight, step_per_sec: s_per_sec}
        summary, _ = sess.run([merged_train, train_op], feed_dict=feed_dict)
        train_writer.add_summary(summary, epoch)

        if epoch % 100 == 0:
            # images, labels, weight = sess.run(test_next_element)

            p = np.random.permutation(len(test_x))[:1500]
            images = test_x[p]
            labels = test_y[p]
            weight = test_w[p]

            feed_dict = {x: images, y: labels, w: weight, step_per_sec: s_per_sec}
            summary, loss, acc = sess.run([merged_test, loss_op, accuracy_op], feed_dict=feed_dict)
            test_writer.add_summary(summary, epoch)

            print(f'epoch: {epoch} loss: {loss:.3f} accuracy: {acc:.3f} s/step: {delta_time:.3f}')

        if epoch % config.CHECKPOINT_INTERVAL == 0:
            saver.save(sess, training_dir + '/model', global_step=epoch)
            saver.export_meta_graph(training_dir + f'/model-{epoch}.meta')

    print('Training Finished!')

    saver.save(sess, training_dir + '/model', global_step=config.EPOCHS)
    saver.export_meta_graph(training_dir + f'/model-{config.EPOCHS}.meta')

    train_writer.close()
    test_writer.close()
