import os

from utils import data_reader, tfrecords
from random import shuffle


def create(img_dirs, out_dir, training_percentage=0.8):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    train_path = os.path.join(out_dir, 'data_train.tfrecord')
    test_path = os.path.join(out_dir, 'data_test.tfrecord')

    data = data_reader.load(img_dirs)
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

    out_dir = './data'
    create(img_dirs, out_dir)


if __name__ == "__main__":
    main()
