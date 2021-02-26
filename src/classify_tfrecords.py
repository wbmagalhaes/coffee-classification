import sys
import argparse
import tensorflow as tf

from utils.tfrecords import read_tfrecord
from utils.visualize import plot_images, plot_confusion_matrix


def classify_tfs(filenames, modeldir, im_size, batch):
    dataset = read_tfrecord(filenames, im_size=im_size)
    x_data, y_true = zip(*[data for data in dataset])

    model = tf.keras.models.load_model(modeldir)
    _, y_pred = model.predict(dataset.batch(batch))

    return x_data, y_true, y_pred


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdir', type=str, default='data/teste_dataset.tfrecord')
    parser.add_argument('-m', '--modeldir', type=str, default='models/CoffeeNet6')
    parser.add_argument('--im_size', type=int, default=64)
    parser.add_argument('--batch', type=int, default=36)
    args = parser.parse_args()

    x, true, pred = classify_tfs(
        filenames=[args.inputdir],
        modeldir=args.modeldir,
        im_size=args.im_size,
        batch=args.batch
    )

    plot_images(x[:args.batch], true[:args.batch], pred[:args.batch], fontsize=8)
    plot_confusion_matrix(true, pred)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
