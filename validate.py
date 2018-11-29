import tensorflow as tf

from utils import config
from utils.tfrecords import get_data
from utils import visualize

export_dir = 'saved_models/simple_4/'

val_x, val_y = get_data([config.VALIDATION_PATH], shuffle=False)

print('Validation data loaded.')

with tf.Session(graph=tf.Graph()) as sess:
    tf.global_variables_initializer().run()

    tf.saved_model.loader.load(sess, ["serve"], export_dir)

    graph = tf.get_default_graph()
    print('Graph restored.')

    print('Starting predictions.')
    feed_dict = {
        'inputs/image_input:0': val_x,
        'inputs/is_training:0': False
    }

    labels, probs = sess.run(
        ['result/label:0', 'result/probs:0'], feed_dict=feed_dict)

    visualize.show_images_compare(
        val_x, labels, probs, val_y, len(val_x), 'validate_result.jpg')
    visualize.show_accuracy(labels, val_y)
    print('Predictions completed!')
