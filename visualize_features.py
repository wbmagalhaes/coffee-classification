import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv

from utils import config
from utils.tfrecords import get_data
from utils import labelmap

model_id = 'CoffeeNet6'
print('Using model', model_id)

export_dir = 'saved_models/' + model_id + '/'

data_x, data_y = get_data([config.VALIDATION_PATH], shuffle=True)
print(len(data_x))
print('Data loaded.')

FILTER_IMG_SIZE = 128


def plotNNFilter(units, title, im_name):
    n = units.shape[3]

    n_columns = int(math.sqrt(n))
    n_rows = int(math.ceil(n / n_columns))

    filters = np.split(units, n, axis=3)

    image = np.zeros((n_rows * FILTER_IMG_SIZE, n_columns * FILTER_IMG_SIZE))

    for i in range(n_rows):
        filter_row = filters[i * n_columns:min((i + 1) * n_columns, n)]

        for j in range(n_columns):
            if j < len(filter_row):
                filter_img = filter_row[j][0]
                img = cv.resize(
                    src=filter_img,
                    dsize=(FILTER_IMG_SIZE, FILTER_IMG_SIZE),
                    interpolation=cv.INTER_NEAREST)

                i_min = i * FILTER_IMG_SIZE
                i_max = (i+1) * FILTER_IMG_SIZE
                j_min = j * FILTER_IMG_SIZE
                j_max = (j+1) * FILTER_IMG_SIZE

                image[i_min:i_max, j_min:j_max] = img

    #plt.title(title)
    #im = plt.imshow(image, cmap='jet')
    #plt.colorbar(im)

    #plt.show()

    image = (image * 255).astype(np.uint8)
    image = cv.applyColorMap(image, cv.COLORMAP_JET)

    cv.imwrite(im_name, image)


with tf.Session(graph=tf.Graph()) as sess:
    tf.global_variables_initializer().run()

    tf.saved_model.loader.load(sess, ["serve"], export_dir)
    graph = tf.get_default_graph()
    print('Graph restored.')

    image_input = graph.get_tensor_by_name('inputs/image_input:0')
    is_training = graph.get_tensor_by_name('inputs/is_training:0')

    layer1 = graph.get_tensor_by_name('neural_net/CONV1/Relu:0')
    layer2 = graph.get_tensor_by_name('neural_net/CONV2/Relu:0')
    layer3 = graph.get_tensor_by_name('neural_net/CONV3/Relu:0')
    layer4 = graph.get_tensor_by_name('neural_net/CONV4/Relu:0')
    layer5 = graph.get_tensor_by_name('neural_net/DENSE5/Relu:0')
    layer6 = graph.get_tensor_by_name('neural_net/DENSE6/BiasAdd:0')

    label = graph.get_tensor_by_name('result/label:0')
    probs = graph.get_tensor_by_name('result/probs:0')

    feed_dict = {
        image_input: data_x,
        is_training: False
    }

    units1, units2, units3, units4, units5, units6, _label, _probs = sess.run(
        [layer1, layer2, layer3, layer4, layer5, layer6, label, probs], feed_dict=feed_dict)

    i = 0
    for y, l, p, u1, u2, u3, u4 in zip(data_y, _label, _probs, units1, units2, units3, units4):
        right = np.argmax(y)

        right_label = labelmap.name_of_idx(right)
        pred_label = labelmap.name_of_idx(l)
        pred_conf = p[l]

        title = '%i-layer1: %s: %s %.2f' % (i, right_label, pred_label, pred_conf)
        im_name = 'features/%s_%i_layer_1.jpg' % (right_label, i)
        plotNNFilter(np.array([u1]), title, im_name)
        
        title = '%i-layer2: %s: %s %.2f' % (i, right_label, pred_label, pred_conf)
        im_name = 'features/%s_%i_layer_2.jpg' % (right_label, i)
        plotNNFilter(np.array([u2]), title, im_name)

        title = '%i-layer3: %s: %s %.2f' % (i, right_label, pred_label, pred_conf)
        im_name = 'features/%s_%i_layer_3.jpg' % (right_label, i)
        plotNNFilter(np.array([u3]), title, im_name)

        title = '%i-layer4: %s: %s %.2f' % (i, right_label, pred_label, pred_conf)
        im_name = 'features/%s_%i_layer_4.jpg' % (right_label, i)
        plotNNFilter(np.array([u4]), title, im_name)

        i += 1
