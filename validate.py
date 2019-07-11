import tensorflow as tf
import numpy as np

from utils import config
from utils import labelmap
from utils.tfrecords import get_data
from collections import defaultdict

from utils import visualize

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

model_id = 'CoffeeNet6'
print('Using model', model_id)

export_dir = 'saved_models/' + model_id + '/'

val_x, val_y = get_data(filenames=[config.VALIDATION_PATH], shuffle=True)
print(len(val_x))
print('Validation data loaded.')

with tf.Session(graph=tf.Graph()) as sess:
    tf.global_variables_initializer().run()
    tf.saved_model.loader.load(sess, ["serve"], export_dir)

    graph = tf.get_default_graph()
    print('Graph restored.')

    predictions = tf.placeholder(tf.int32, [None])
    real_labels = tf.placeholder(tf.int32, [None])
    confusion_matrix = tf.confusion_matrix(labels=real_labels, predictions=predictions)

    print('Starting predictions.')
    feed_dict = {
        'inputs/img_input:0': val_x
    }

    labels, probs = sess.run(['result/label:0', 'result/probs:0'], feed_dict=feed_dict)

    img_list = []
    val_list = []
    correct_list = []
    label_list = []
    confidence_list = []
    for x, y, label, prob in zip(val_x, val_y, labels, probs):
        right = np.argmax(y)

        val_list.append(right)

        if right != label:
            img_list.append(x)
            confidence_list.append(prob)
            correct_list.append(y)
            label_list.append(label)

            error_type = ''
            if (right > label):
                error_type = '%s-%s' % (
                    labelmap.name_of_idx(right),
                    labelmap.name_of_idx(label))
            else:
                error_type = '%s-%s' % (
                    labelmap.name_of_idx(label),
                    labelmap.name_of_idx(right))

    matrix = sess.run([confusion_matrix], feed_dict={real_labels: val_list, predictions: labels})

    df_cm = pd.DataFrame(matrix[0],
                         index=[i['name'] for i in labelmap.labels],
                         columns=[i['name'] for i in labelmap.labels])

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()

    visualize.show_images_compare(img_list, label_list, confidence_list, correct_list, len(correct_list), 'validate_result.jpg')
    
    visualize.show_accuracy(labels, val_y)
    print('Predictions completed!')