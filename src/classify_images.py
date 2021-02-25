from utils.segmentation import segment_images, crop_beans


def open_images(images_dir):
    images, beans_data = segment_images(images_dir)

    dataset = []
    for image, data in zip(images, beans_data):
        beans = crop_beans(image, data, cut_size=64, bg_color=(0, 0, 0))
        dataset.extend(beans)

    return dataset


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imagesdir', type=str, default='images')
    args = parser.parse_args()

    dataset = open_images(args.imagesdir)

    # classify
    # show result


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
