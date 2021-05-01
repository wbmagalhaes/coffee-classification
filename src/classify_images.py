import os
import sys
import argparse

import numpy as np
import tensorflow as tf

from utils.data_reader import open_images, open_json
from utils.segmentation import process_image, crop_beans, count_beans_pred


def load_images(images_dir, im_size=64, load_previous=True):
    images, addrs = open_images(images_dir)

    dataset = []

    for image, addr in zip(images, addrs):
        json_addr = addr[:-3] + 'json'

        if os.path.isfile(json_addr) and load_previous:
            data = open_json(json_addr)
        else:
            data = process_image(image)

        beans_data = crop_beans(image, data, cut_size=im_size)
        image_data = [crop.astype(np.float32) / 255. for crop, _ in beans_data]
        dataset.append(image_data)

    return addrs, dataset


def classify_imgs(dataset, modeldir):
    model = tf.keras.models.load_model(modeldir)
    return [classify_img(data, model) for data in dataset]


def classify_img(data, model):
    data = np.array(data)
    _, pred = model.predict(data)
    return pred


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imagesdir', type=str, default='images')
    parser.add_argument('-m', '--modeldir', type=str, default='models/saved_models/CoffeeNet6')
    parser.add_argument('--ignore', dest='ignore_previous', action='store_true', default=False)
    parser.add_argument('--im_size', type=int, default=64)
    args = parser.parse_args()

    addrs, dataset = load_images(args.imagesdir, args.im_size, (not args.ignore_previous))
    preds = classify_imgs(dataset, args.modeldir)

    for pred, addr in zip(preds, addrs):
        print(addr)
        count_beans_pred(pred)
        print('')


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
