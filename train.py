import tensorflow as tf
import numpy as np
import time
import glob

from utils import config
from utils import labelmap
from utils.tfrecords import get_data
from utils.data_augment import aug_data

from utils.model import loss_function
from utils.model import accuracy_function

import CoffeeNet6 as cnn

training_dir = config.CHECKPOINT_DIR + cnn.model_id

print('Using model', cnn.model_id)

with tf.name_scope('dataset_load'):
    train_x, train_y = get_data(filenames=[config.TRAINING_PATH], shuffle=True)
    test_x, test_y = get_data(filenames=[config.TESTING_PATH], shuffle=True)

with tf.name_scope('inputs'):
    x = tf.placeholder(
        tf.uint8,
        [None, config.IMG_SIZE, config.IMG_SIZE, 3],
        name='image_input'
    )
    y = tf.placeholder(
        tf.float32,
        [None, labelmap.count],
        name='label_input'
    )
    is_training = tf.placeholder(tf.bool, name='is_training')

    augument_op = aug_data(x)

with tf.name_scope('neural_net'):
    y_pred = cnn.model(x, is_training)

with tf.name_scope('result'):
    label = tf.argmax(y_pred, 1, name='label')
    probs = tf.nn.softmax(y_pred, name='probs')

with tf.name_scope('score'):
    y_true = tf.argmax(y, 1)
    loss_op = loss_function(y_pred, y_true)
    accuracy_op = accuracy_function(label, y_true)

tf.summary.scalar('score/loss', loss_op)
tf.summary.scalar('score/accuracy', accuracy_op)

global_step = tf.train.get_or_create_global_step()

learning_rate = tf.train.exponential_decay(
    learning_rate=config.LEARNING_RATE,
    global_step=global_step,
    decay_steps=1000,
    decay_rate=0.94,
    staircase=False
)

tf.summary.scalar('learning_rate', learning_rate)

step_per_sec = tf.placeholder(tf.float32)
tf.summary.scalar('step_per_sec', step_per_sec)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate, name='AdamOpt')
    train_op = optimizer.minimize(
        loss_op, global_step=global_step, name='TrainOp')

merged = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(training_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(training_dir + '/test')

    tf.global_variables_initializer().run()

    time_i = time.time()

    print('Starting train...')
    for epoch in range(config.EPOCHS + 1):
        delta_time = time.time() - time_i
        time_i = time.time()

        if delta_time <= 0:
            delta_time = 1
        s_per_sec = 1.0 / delta_time

        p = np.random.permutation(len(train_x))[:config.BATCH_SIZE]
        batch_x = train_x[p]
        batch_y = train_y[p]

        aug_x = sess.run(augument_op, feed_dict={x: batch_x})

        feed_dict = {x: aug_x, y: batch_y,
                     step_per_sec: s_per_sec, is_training: True}
        summary, _ = sess.run([merged, train_op], feed_dict=feed_dict)
        train_writer.add_summary(summary, epoch)

        if epoch % 10 == 0:
            feed_dict = {x: test_x, y: test_y,
                         step_per_sec: s_per_sec, is_training: False}
            summary, loss, acc = sess.run(
                [merged, loss_op, accuracy_op], feed_dict=feed_dict)

            test_writer.add_summary(summary, epoch)

            print(
                'epoch: {} loss: {:.3f} accuracy: {:.3f} s/step: {:.3f}'.format(
                    epoch, loss, acc, delta_time))

        if epoch % config.CHECKPOINT_INTERVAL == 0:
            saver.save(sess, training_dir + '/model', global_step=epoch)
            saver.export_meta_graph(
                training_dir + '/model-{}.meta'.format(epoch))

    saver.save(sess, training_dir + '/model', global_step=config.EPOCHS)
    saver.export_meta_graph(
        training_dir + '/model-{}.meta'.format(config.EPOCHS))

    train_writer.close()
    test_writer.close()

print('Training Finished!')
