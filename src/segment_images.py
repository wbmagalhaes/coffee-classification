import sys
import argparse

from utils.data_reader import open_images


def segment(images_dir, overwrite=False):
    images, addrs = open_images(images_dir)

    # for image in images:
        # get segment
        # give json addr

    return [segment_image(image, addr, overwrite) for image, addr in zip(images, addrs)]


def save_segmentation(imgs_data, output_dir=None):
    for img in imgs_data:
        if imgs_data[img]['data']:

            if output_dir:
                path = os.path.join(output_dir, img)
            else:
                path = imgs_data[img]['path']

            with open(path, 'w+') as f:
                json.dump(imgs_data[img]['data'], f, indent=2)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imagesdir', type=str, default='images')
    parser.add_argument('-o', '--outputdir', type=str, default=None)
    args = parser.parse_args()

    imgs_data = segment(args.imagesdir)
    save_segmentation(imgs_data, args.outputdir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
