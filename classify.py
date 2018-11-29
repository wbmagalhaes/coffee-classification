import tensorflow as tf
import glob

from utils import config
from utils import labelmap
from utils.data_reader import read_xml
from utils import visualize

export_dir = 'saved_models/simple_4/'

with tf.Session(graph=tf.Graph()) as sess:
    tf.global_variables_initializer().run()

    tf.saved_model.loader.load(sess, ["serve"], export_dir)

    graph = tf.get_default_graph()
    print('Graph restored.')

    print('Starting predictions.')

    for addr in glob.glob('result/*.xml'):
        filename, imgs, _ = read_xml('result', addr)
        feed_dict = {
            'inputs/image_input:0': imgs,
            'inputs/is_training:0': False
        }

        labels, probs = sess.run(
            ['result/label:0', 'result/probs:0'],
            feed_dict=feed_dict)

        visualize.show_images(imgs, labels, probs, len(
            imgs), 'result/out_{}'.format(filename))

        defects = 0
        count = [0] * labelmap.count
        for label in labels:
            count[label] += 1
            defects += labelmap.labels[label]['weight']

        print(' ')
        print('==================')
        for i in range(labelmap.count):
            print(labelmap.labels[i]['name'], count[i])
        print('Defects: {:.2f}'.format(defects))
        print('==================')

    print('Predictions completed!')
