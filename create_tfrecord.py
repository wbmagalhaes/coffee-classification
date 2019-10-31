import tensorflow as tf

import os
import data_reader
import tfrecords

from random import shuffle


def from_dirs(in_dirs, out_dir):
    training_percentage = 0.8

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    train_path = os.path.join(out_dir, 'data_train.tfrecord')
    test_path = os.path.join(out_dir, 'data_test.tfrecord')

    data = data_reader.load(in_dirs)
    shuffle(data)

    train_num = int(len(data) * training_percentage)

    train_data = data[:train_num]
    test_data = data[train_num:]

    print(f'{len(train_data)} train images.')
    print(f'{len(test_data)} test images.')

    print('Writing tfrecords...')
    tfrecords.write(train_path, train_data)
    tfrecords.write(test_path, test_data)
    print('Finished.')


def main():
    img_dirs = [
        'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs0',
        'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs1',
        'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs2',
        'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs3',
        'C:/Users/Usuario/Desktop/cafe_imgs/cut_imgs4'
    ]

    data_dir = './data'

    from_dirs(img_dirs, data_dir)


if __name__ == "__main__":
    main()
