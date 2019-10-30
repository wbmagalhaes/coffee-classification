import os
import csv
import cv2

import numpy as np

from random import shuffle

from utils.labelmap import label_names


def read_csv(data_dir, csv_name):
    csv_path = os.path.join(data_dir, csv_name)
    with open(csv_path, 'r') as readfile:
        reader = csv.reader(readfile)
        lines = list(reader)[1:]

    return lines


def open_img(data_dir, img_data):
    img_name = img_data[1]
    img_label = img_data[2]

    img_path = os.path.join(data_dir, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    label = label_names.index(img_label)
    return img, label


def load(dirs, csv_name='coffee_data.csv'):
    data = []
    for data_dir in dirs:
        print(f'Loading data from: {data_dir}')
        lines = read_csv(data_dir, csv_name)
        csv_data = [open_img(data_dir, line) for line in lines]
        data.extend(csv_data)

    print(f'Data loaded. {len(data)} images.')
    return data


def split_train(data, percentage=0.8):
    shuffle(data)
    x, y = zip(*data)

    print(x[0])
    print(y[0])

    # x = np.array(x)
    # y = np.array(y)

    #x = x.astype(np.float32)
    y = y.astype(np.float32)

    train_num = int(len(data) * percentage)

    x_train = x[:train_num]
    y_train = y[:train_num]

    x_test = x[train_num:]
    y_test = y[train_num:]

    print(f'{len(x_train)} train images.')
    print(f'{len(x_test)} test images.')

    return (x_train, y_train), (x_test, y_test)
