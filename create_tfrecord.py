import tensorflow as tf

from utils import data_reader, tfrecords

tf.enable_eager_execution()

img_dirs = [
    'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs0',
    'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs1',
    'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs2',
    'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs3',
    'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs4'
]

out_path = './data/coffee_data.tfrecord'

data = data_reader.load(img_dirs)

print('Writing tfrecord...')
tfrecords.write_tfrecord(out_path, data)
print('Finished.')
