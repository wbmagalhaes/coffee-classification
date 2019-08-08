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

data_x, data_y = get_data(filenames=[config.VALIDATION_PATH], shuffle=True)
print(f'Data loaded: {len(data_x)}')


def plotNNFilter(units, title, im_name, filter_img_size=128):
    n = units.shape[3]

    n_columns = int(math.sqrt(n))
    n_rows = int(math.ceil(n / n_columns))

    filters = np.split(units, n, axis=3)

    image = np.zeros((n_rows * filter_img_size, n_columns * filter_img_size))

    for i in range(n_rows):
        filter_row = filters[i * n_columns:min((i + 1) * n_columns, n)]

        for j in range(n_columns):
            if j < len(filter_row):
                filter_img = filter_row[j][0]
                img = cv.resize(
                    src=filter_img,
                    dsize=(filter_img_size, filter_img_size),
                    interpolation=cv.INTER_NEAREST)

                i_min = i * filter_img_size
                i_max = (i+1) * filter_img_size
                j_min = j * filter_img_size
                j_max = (j+1) * filter_img_size

                image[i_min:i_max, j_min:j_max] = img

    # plt.title(title)
    # im = plt.imshow(image, cmap='jet')
    # plt.colorbar(im)
    # plt.show()

    image = (image * 255).astype(np.uint8)
    image = cv.applyColorMap(image, cv.COLORMAP_JET)
    
    print(im_name)
    cv.imwrite(im_name, image)


with tf.Session(graph=tf.Graph()) as sess:
    tf.global_variables_initializer().run()

    tf.saved_model.loader.load(sess, ["serve"], export_dir)
    graph = tf.get_default_graph()
    print('Graph restored.')

    for op in graph.get_operations():
        print(op.name)

    image_input = graph.get_tensor_by_name('inputs/img_input:0')

    layer1 = graph.get_tensor_by_name('neural_net/CONV1/CONV1/LeakyRelu:0')
    layer2 = graph.get_tensor_by_name('neural_net/CONV2/CONV2/LeakyRelu:0')
    layer3 = graph.get_tensor_by_name('neural_net/CONV3/CONV3/LeakyRelu:0')
    layer4 = graph.get_tensor_by_name('neural_net/CONV4/CONV4/LeakyRelu:0')

    label = graph.get_tensor_by_name('result/label:0')
    probs = graph.get_tensor_by_name('result/probs:0')

    units1, units2, units3, units4, _label, _probs = sess.run(
        [layer1, layer2, layer3, layer4, label, probs], feed_dict={image_input: data_x})

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
