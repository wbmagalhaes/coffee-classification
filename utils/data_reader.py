import os
import glob
import json
import cv2

import numpy as np

from utils.labelmap import label_names


def read_json(addr, cut_size, bg_color):
    with open(addr, 'r') as json_file:
        data = json.load(json_file)

    dirname = os.path.dirname(addr)
    filename = data['imagePath']

    img_path = os.path.join(dirname, filename)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    shapes = data['shapes']

    return [cut_shape(image, shape, cut_size, bg_color) for shape in shapes]


def cut_shape(image, shape, cut_size, bg_color):
    image = image.copy()
    im_h, im_w, _ = image.shape

    img_label = shape['label']
    label = label_names.index(img_label)

    points = shape['points']
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

    # cv2.imshow('cropped', cropped)
    # cv2.waitKey()

    return cropped, label


def load(dirs, cut_size=64, bg_color=(255, 0, 0)):
    data = []
    for data_dir in dirs:
        print(f'Loading data from: {data_dir}')

        addrs = glob.glob(os.path.join(data_dir, '*.json'))
        for addr in addrs:
            json_data = read_json(addr, cut_size, bg_color)
            data.extend(json_data)

    print(f'Data loaded. {len(data)} images.')
    return data
