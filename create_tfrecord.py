import sys
import argparse

from utils.tfrecords import create_dataset


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputdir', type=str, default='./images')
    parser.add_argument('-o', '--outputdir', type=str, default='./data')
    parser.add_argument('--train', type=float, default=0.8)
    parser.add_argument('--splits', nargs='+', type=int, default=(1, 1, 1))

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--shuffle', dest='random', action='store_true')
    feature_parser.add_argument('--no-shuffle', dest='random', action='store_false')
    parser.set_defaults(random=True)

    args = parser.parse_args()

    create_dataset(
        input_dir=args.inputdir,
        output_dir=args.outputdir,
        training_percentage=args.train,
        random=args.random,
        splits=args.splits
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
