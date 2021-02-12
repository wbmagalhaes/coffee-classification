import os
import glob
import json
import cv2

import numpy as np

from utils.labelmap import label_names
import matplotlib.pyplot as plt


def read_json(addr, cut_size, bg_color):
    with open(addr, 'r') as json_file:
        data = json.load(json_file)

    dirname = os.path.dirname(addr)
    filename = os.path.basename(addr)[:-4] + "jpg"

    img_path = os.path.join(dirname, filename)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return [cut_bean(image, bean, cut_size, bg_color) for bean in data]


def cut_bean(image, bean, cut_size, bg_color):
    image = image.copy()
    im_h, im_w, _ = image.shape

    img_label = bean['label']
    label = label_names.index(img_label)

    points = bean['points']
    xs, ys = zip(*points)

    xmin = int(max(min(xs), 0))
    ymin = int(max(min(ys), 0))
    xmax = int(min(max(xs), im_w - 1))
    ymax = int(min(max(ys), im_h - 1))

    size_x = xmax - xmin
    size_y = ymax - ymin
    size = max(size_x, size_y) // 2

    center_x = int(size_x / 2) + xmin
    center_y = int(size_y / 2) + ymin

    xmin = int(max(center_x - size, 0))
    ymin = int(max(center_y - size, 0))
    xmax = int(min(center_x + size, im_w - 1))
    ymax = int(min(center_y + size, im_h - 1))

    mask = image.copy()
    mask *= 0

    points = np.array(points, dtype=np.int32)

    cv2.fillPoly(mask, [points], (1., 1., 1.))
    mask = mask[ymin:ymax, xmin:xmax]

    cropped = image[ymin:ymax, xmin:xmax].astype(np.float32)
    cropped = cropped * mask

    bg = mask.copy()
    bg[:, :, :] = bg_color
    bg = bg * (1 - mask)

    cropped = (cropped + bg).astype(np.uint8)
    cropped = cv2.resize(cropped, dsize=(cut_size, cut_size))

    return cropped, label


def load(data_dir, cut_size=64, bg_color=(255, 0, 0)):
    count = {}
    data = []

    addrs = glob.glob(data_dir + '/**/*.json', recursive=True)
    for addr in addrs:
        print(os.path.basename(addr))

        beans = read_json(addr, cut_size, bg_color)
        data.extend(beans)

        for bean in beans:
            _, label = bean

            if label not in count.keys():
                count[label] = 0
            count[label] += 1

    for key in count.keys():
        print(f'{label_names[key]}: {count[key]}')

    return data


def count_beans_in_list(bean_list):
    count = {}
    for bean in bean_list:
        _, label = bean

        if label not in count.keys():
            count[label] = 0
        count[label] += 1

    for key in count.keys():
        print(f'{label_names[key]}: {count[key]}')
