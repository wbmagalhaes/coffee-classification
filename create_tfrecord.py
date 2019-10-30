import tensorflow as tf

from utils import data_reader, tfrecords

from random import shuffle

tf.enable_eager_execution()

img_dirs = [
    'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs0',
    'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs1',
    'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs2',
    'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs3',
    'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs4'
]

train_path = './data/data_train.tfrecord'
test_path = './data/data_test.tfrecord'

data = data_reader.load(img_dirs)
shuffle(data)

train_num = int(len(data) * 0.8)

train_data = data[:train_num]
test_data = data[train_num:]

print(f'{len(train_data)} train images.')
print(f'{len(test_data)} test images.')

print('Writing tfrecords...')
tfrecords.write_tfrecord(train_path, train_data)
tfrecords.write_tfrecord(test_path, test_data)
print('Finished.')
