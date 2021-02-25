import sys
import argparse

from utils.tfrecords import read_tfrecord
from utils.visualize import plot_images, plot_confusion_matrix
from utils.reload_model import from_savedmodel


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputdir', type=str, default='data/teste_dataset.tfrecord')
    parser.add_argument('-m', '--modeldir', type=str, default='models/CoffeeNet6')

    parser.add_argument('--batch', type=int, default=36)

    args = parser.parse_args()

    dataset = read_tfrecord([args.inputdir])
    x_data, y_true = zip(*[data for data in dataset])

    model = from_savedmodel(args.modeldir)
    _, y_pred = model.predict(dataset.batch(args.batch))

    plot_images(x_data[:args.batch], y_true[:args.batch], y_pred[:args.batch], fontsize=8)
    plot_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
