import tensorflow as tf
import glob

import xml.etree.ElementTree as ET

from utils import config
from utils import labelmap
from utils.data_reader import read_xml
from utils import visualize

model_id = 'CoffeeNet6'
print('Using model', model_id)

export_dir = 'saved_models/' + model_id + '/'
imgs_dir = 'result'


def update_xml(addr, labels):
    # open xml
    tree = ET.parse(addr)
    root = tree.getroot()

    # get objs list
    objs = root.findall('object')

    # change objs list
    for (obj, label) in zip(objs, labels):
        obj.find('name').text = labelmap.name_of_idx(label)

    # write xml
    tree.write(addr)


with tf.Session(graph=tf.Graph()) as sess:
    tf.global_variables_initializer().run()
    tf.saved_model.loader.load(sess, ["serve"], export_dir)

    graph = tf.get_default_graph()
    print('Graph restored.')

    print('Starting predictions.')
    print('==================')

    erros = 0
    total = 0
    for addr in glob.glob(imgs_dir + '/*.xml'):
        filename, imgs, real_ys = read_xml(imgs_dir, addr)

        feed_dict = {
            'inputs/image_input:0': imgs,
            'inputs/is_training:0': False
        }
        labels, probs = sess.run(
            ['result/label:0', 'result/probs:0'],
            feed_dict=feed_dict
        )

        visualize.show_images(imgs, labels, probs, len(imgs), imgs_dir + '/out_{}'.format(filename))

        pred_defects = 0
        real_defects = 0
        pred_count = [0] * labelmap.count
        real_count = [0] * labelmap.count
        for pred_label, real_label in zip(labels, real_ys):
            if real_label == -1:
                continue

            if pred_label != real_label:
                erros += 1

            pred_count[pred_label] += 1
            real_count[real_label] += 1

            pred_defects += labelmap.labels[pred_label]['weight']
            real_defects += labelmap.labels[real_label]['weight']

        total += len(real_ys)

        for label, pred_c, real_c in zip(labelmap.labels, pred_count, real_count):
            print('{:<12s} {:>5} / {:}'.format(label['name'], pred_c, real_c))

        # update_xml(addr, labels)

        print('{:<12s} {:>5.1f} / {:.1f}'.format('defeitos', pred_defects, real_defects))
        print('==================')

    print('Predictions completed!')

    acc = (total - erros) / total
    print('Accuracy: {:.2f}%'.format(acc * 100))
