import tensorflow as tf
import numpy as np
import math

from utils import config
from utils import labelmap
from utils.tfrecords import get_iterator

import matplotlib.pyplot as plt

model_id = 'CoffeeNet5_gap'
export_dir = 'saved_models/' + model_id + '/'
save_dir = 'saved_features/' + model_id + '/'

layer = 'CONV2'

names = [label['name'] for label in labelmap.labels]

print(f'Using model {model_id}')

with tf.Session(graph=tf.Graph()) as sess:
    sess.run(tf.global_variables_initializer())

    tf.saved_model.loader.load(sess, ["serve"], export_dir)
    graph = tf.get_default_graph()
    print('Graph restored.')

    for op in graph.get_operations():
        print(op.name)

    img_input = graph.get_tensor_by_name('inputs/img_input:0')
    output = graph.get_tensor_by_name(f'neural_net/{layer}/LeakyRelu:0')

    val_iter, val_next_element = get_iterator([config.VALIDATION_PATH], batch_size=30)
    sess.run(val_iter.initializer)

    x, y, _ = sess.run(val_next_element)
    outputs = sess.run(output, feed_dict={img_input: x})

    for n in range(len(outputs)):
        label = np.argmax(y[n])
        out = outputs[n]

        name = f'{layer} - {names[label]}'

        print(name)
        print(out.shape)

        h, w, f = out.shape
        grid_n = int(math.sqrt(f))

        fig, axes = plt.subplots(grid_n, grid_n)
        fig.suptitle(name)

        for i, ax in enumerate(axes.flat):
            img = out[:, :,  i]
            img = (img - np.min(img)) / np.ptp(img)

            ax.imshow(img, interpolation='bicubic', cmap='jet')

            ax.set_xticks([])
            ax.set_yticks([])

        # plt.show()

        plt.savefig(f'{save_dir}{name}.png', bbox_inches='tight')
