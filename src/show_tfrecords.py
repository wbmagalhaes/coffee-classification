import sys
import argparse

from utils.tfrecords import read_tfrecord, show_dataset


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='data/valid_dataset.tfrecord')
    parser.add_argument('--batch', type=int, default=36)
    parser.add_argument('--augment', dest='augment', action='store_true', default=False)
    args = parser.parse_args()

    dataset = read_tfrecord([args.path])

    show_dataset(
        dataset=dataset,
        batch=args.batch,
        augment=args.augment,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
