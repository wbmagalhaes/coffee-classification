import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from utils import config
from utils import labelmap
from utils.tfrecords import get_iterator

from tensorflow.contrib.tensorboard.plugins import projector

import cv2

model_id = 'CoffeeNet6_gap_dense_64'
export_dir = 'saved_models/' + model_id + '/'
pca_dir = 'pca_visualizer/' + model_id + '/'

print(f'Using model {model_id}')

with tf.Session(graph=tf.Graph()) as sess:
    sess.run(tf.global_variables_initializer())

    tf.saved_model.loader.load(sess, ["serve"], export_dir)
    graph = tf.get_default_graph()
    print('Graph restored.')

    data_iter, data_next_element = get_iterator([config.VALIDATION_PATH, config.TESTING_PATH], batch_size=1000)
    sess.run(data_iter.initializer)

    print('Starting predictions...')

    true_labels = []
    pred_logits = []

    while True:
        try:
            images, true, _ = sess.run(data_next_element)

            feed_dict = {'inputs/img_input:0': images}
            logits = sess.run('result/logits:0', feed_dict=feed_dict)

            true_labels.extend(true)

            for logit in logits:
                pred_logits.append(logit)

        except tf.errors.OutOfRangeError:
            break

    summary_writer = tf.summary.FileWriter(pca_dir)
    embedding_var = tf.Variable(np.array(pred_logits), name='logits_embedding')

    p_config = projector.ProjectorConfig()
    embedding = p_config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    embedding.metadata_path = 'metadata.tsv'
    # embedding.sprite.image_path = 'sprite.png'
    # embedding.sprite.single_image_dim.extend([28, 28])

    projector.visualize_embeddings(summary_writer, p_config)

    tf.global_variables_initializer().run()

    # save logits
    saver = tf.train.Saver()
    saver.save(sess, pca_dir + 'model.ckpt')
    # saver.export_meta_graph(pca_dir + '/model.meta')

    # save labels
    with open(pca_dir + embedding.metadata_path, 'w') as meta:
        meta.write('Index\tLabel\n')
        for index, label in enumerate(true_labels):
            name = labelmap.name_of_idx(np.argmax(label))
            meta.write(f'{index}\t{name}\n')

    # save sprites
    # rows = 28
    # cols = 28
    # sprite_dim = int(np.sqrt(val_x.shape[0]))
    # sprite_image = np.ones((cols * sprite_dim, rows * sprite_dim, 3))

    # index = 0
    # labels = []
    # for i in range(sprite_dim):
    #     for j in range(sprite_dim):
    #         img = cv2.resize(val_x[index], (rows, cols))

    #         sprite_image[i * cols: (i + 1) * cols, j * rows: (j + 1) * rows, :] = img
    #         index += 1

    # plt.imsave(pca_dir + embedding.sprite.image_path, sprite_image)
    # plt.imshow(sprite_image, cmap='gray')
    # plt.show()

    print('Predictions completed!')
