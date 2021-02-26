import os
import sys
import argparse

from utils.data_reader import open_images, open_json, save_json
from utils.segmentation import process_image


def make_segmentation(images_dir, load_previous=True, output_dir=None):
    images, addrs = open_images(images_dir)

    json_addrs = []
    imgs_data = []

    for image, addr in zip(images, addrs):
        json_addr = addr[:-3] + 'json'

        if output_dir:
            json_addr = os.path.join(output_dir, os.path.basename(json_addr))

        if os.path.isfile(json_addr) and load_previous:
            img_data = open_json(json_addr)
            print(f'{json_addr} já existe.')
        else:
            img_data = process_image(image)

        json_addrs.append(json_addr)
        imgs_data.append(img_data)

    return json_addrs, imgs_data


def save_segmentation(json_addrs, imgs_data, overwrite=False):
    for json_addr, img_data in zip(json_addrs, imgs_data):
        if overwrite or not os.path.isfile(json_addr):
            save_json(img_data, json_addr)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imagesdir', type=str, default='images')
    parser.add_argument('-o', '--outputdir', type=str, default=None)
    parser.add_argument('--ignore', dest='ignore_previous', action='store_true', default=False)
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', default=False)
    args = parser.parse_args()

    json_addrs, imgs_data = make_segmentation(args.imagesdir, (not args.ignore_previous), args.outputdir)
    save_segmentation(json_addrs, imgs_data, args.overwrite)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
