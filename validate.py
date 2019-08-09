import tensorflow as tf
import numpy as np

from utils import config
from utils import labelmap
from utils.tfrecords import get_data
from collections import defaultdict

from utils import visualize

from pycm import ConfusionMatrix

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

model_id = 'CoffeeNet6_even_more_images'
print('Using model', model_id)

export_dir = 'saved_models/' + model_id + '/'

val_x, val_y = get_data(filenames=[config.VALIDATION_PATH], shuffle=True)
print(f'Validation data loaded: {len(val_x)}')


with tf.Session(graph=tf.Graph()) as sess:
    tf.global_variables_initializer().run()

    tf.saved_model.loader.load(sess, ["serve"], export_dir)

    graph = tf.get_default_graph()
    print('Graph restored.')

    print('Starting predictions...')
    feed_dict = {
        'inputs/img_input:0': val_x
    }

    labels, probs = sess.run(['result/label:0', 'result/probs:0'], feed_dict=feed_dict)

    print('Predictions completed!')

    y_actu = [np.argmax(y) for y in val_y]
    names = [label['name'] for label in labelmap.labels]
    cm = ConfusionMatrix(actual_vector=y_actu, predict_vector=labels)

    print('')

    print('===Discriminant power===')
    dpi = cm.DPI
    for i, label in enumerate(names):
        print(f'{dpi[i]}')

    print('')

    print('===Sensitivity (TPR)===')
    tpr = cm.TPR
    for i, label in enumerate(names):
        print(f'{tpr[i] * 100:.2f}%')

    print('')

    print('===Specificity (TNR)===')
    tnr = cm.TNR
    for i, label in enumerate(names):
        print(f'{tnr[i] * 100:.2f}%')

    print('')

    print('===Positive likelihood ratio===')
    plri = cm.PLRI
    for i, label in enumerate(names):
        print(f'{plri[i]}')

    print('')

    print('===Negative likelihood ratio===')
    nlri = cm.NLRI
    for i, label in enumerate(names):
        print(f'{nlri[i]}')

    print('')

    print('===F1===')
    f1 = cm.F1
    for i, label in enumerate(names):
        print(f'{label}: {f1[i]:.3f}')

    print('')

    print(f'Overall ACC: {cm.Overall_ACC * 100:.2f}%')

    print('')

    print(f'RCI: {cm.RCI:.2f}')
    print(f'SOA1: {cm.SOA1}')
    print(f'SOA2: {cm.SOA2}')
    print(f'SOA3: {cm.SOA3}')
    print(f'SOA4: {cm.SOA4}')
    print(f'SOA5: {cm.SOA5}')
    print(f'SOA6: {cm.SOA6}')

    visualize.plot_confusion_matrix(cm, names)