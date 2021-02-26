import os
import sys
import argparse

from utils.data_reader import open_images, open_json
from utils.segmentation import process_image, crop_beans


def load_images(images_dir, im_size=64, load_previous=False):
    images, addrs = open_images(images_dir)

    dataset = []

    for image, addr in zip(images, addrs):
        json_addr = addr[:-3] + 'json'

        if os.path.isfile(json_addr) and load_previous:
            data = open_json(json_addr)
        else:
            data = process_image(image)

        beans = crop_beans(image, data, cut_size=im_size, bg_color=(0, 0, 0))
        dataset.append(beans)

    return dataset


def classify_imgs(data, modeldir):
    model = tf.keras.models.load_model(modeldir)
    _, y_pred = model.predict(data)
    return y_pred


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imagesdir', type=str, default='images')
    args = parser.parse_args()

    # load images
    dataset = load_images(args.imagesdir, im_size=64, load_previous=False)

    # classify
    classify_imgs()

    # show result
    print(y_pred)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
