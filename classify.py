import tensorflow as tf
import glob

from utils import config
from utils import labelmap
from utils.data_reader import read_xml
from utils import visualize

model_id = 'CoffeeNet6'
print('Using model', model_id)

export_dir = 'saved_models/' + model_id + '/'

with tf.Session(graph=tf.Graph()) as sess:
    tf.global_variables_initializer().run()

    tf.saved_model.loader.load(sess, ["serve"], export_dir)

    graph = tf.get_default_graph()
    print('Graph restored.')

    print('Starting predictions.')

    for addr in glob.glob('result/*.xml'):
        filename, imgs, real_y = read_xml('result', addr)
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
        for label in real_y:
            count[label] += 1
            defects += labelmap.labels[label]['weight']

        #print('==================')
        #for i in range(labelmap.count):
            #print('{}\t{}'.format(labelmap.labels[i]['name'], count[i]))

        print('Defects:\t{:.2f}'.format(defects))
        #print('==================')
        #print(' ')

    print('Predictions completed!')
