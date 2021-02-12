import sys
import argparse

from utils.tfrecords import show_dataset


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='./data/teste_dataset0.tfrecord')
    parser.add_argument('--batch', type=int, default=36)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--augment', dest='augment', action='store_true')
    feature_parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.set_defaults(augment=True)

    args = parser.parse_args()

    show_dataset(
        path=args.path,
        batch=args.batch,
        augment=args.augment,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
