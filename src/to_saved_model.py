import sys
import argparse

import tensorflow as tf
from utils.reload_model import from_json


def export_savedmodel(modeldir, epoch, output):
    model = from_json(modeldir, epoch)

    tf.keras.models.save_model(
        model,
        filepath=output,
        overwrite=True,
        include_optimizer=False,
        save_format='tf'
    )


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', type=str, default='models/CoffeeNet6')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--output', type=int, default='CoffeeNet6')
    args = parser.parse_args()

    export_savedmodel(args.modeldir, args.epoch, args.output)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
