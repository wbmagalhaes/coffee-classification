import tensorflow as tf
import numpy as np

from utils import config
from utils import labelmap
from utils.tfrecords import get_iterator
from collections import defaultdict

from utils import visualize

from pycm import ConfusionMatrix

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

model_id = 'CoffeeNet5_gap_dense'
export_dir = 'saved_models/' + model_id + '/'
print(f'Using model {model_id}')

with tf.Session(graph=tf.Graph()) as sess:
    sess.run(tf.global_variables_initializer())

    tf.saved_model.loader.load(sess, ["serve"], export_dir)
    graph = tf.get_default_graph()
    print('Graph restored.')

    val_iter, val_next_element = get_iterator([config.VALIDATION_PATH], batch_size=1000)
    sess.run(val_iter.initializer)

    print('Starting predictions...')

    true_labels = []
    pred_labels = []

    while True:
        try:
            images, true, _ = sess.run(val_next_element)

            feed_dict = {'inputs/img_input:0': images}
            pred = sess.run('result/label:0', feed_dict=feed_dict)

            true_labels.extend(true)
            pred_labels.extend(pred)

        except tf.errors.OutOfRangeError:
            break

    print('Predictions completed!')

    true_labels = [np.argmax(y) for y in true_labels]
    names = [label['name'] for label in labelmap.labels]
    cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels)

    print('')

    print(f'Overall Accuracy: {cm.Overall_ACC * 100:.2f}%')

    print('===Accuracy===')
    acc = cm.ACC
    for i, label in enumerate(names):
        print(f'{acc[i]:.3f}')

    print('===Precision (Positive predictive value)===')
    pvv = cm.PPV
    for i, label in enumerate(names):
        print(f'{pvv[i]:.3f}')

    print('===Recall (True Positives)===')
    tpr = cm.TPR
    for i, label in enumerate(names):
        print(f'{tpr[i]:.3f}')

    print('===F1===')
    f1 = cm.F1
    for i, label in enumerate(names):
        print(f'{f1[i]:.3f}')

    visualize.plot_confusion_matrix(cm, names, normalize=False)
