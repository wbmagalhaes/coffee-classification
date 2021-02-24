import sys
import argparse

from src.utils.tfrecords import show_dataset


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('--batch', type=int, default=36)
    parser.add_argument('--augment', dest='augment', action='store_true', default=False)

    args = parser.parse_args()

    show_dataset(
        path=args.path,
        batch=args.batch,
        augment=args.augment,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
