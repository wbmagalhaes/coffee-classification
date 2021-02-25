import tensorflow as tf

from utils import reload_model


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--modeldir', type=str, default='models/CoffeeNet6')
    parser.add_argument('--epoch', type=int, default=500)

    args = parser.parse_args()

    model = reload_model.from_json(args.modeldir, args.epoch)

    tf.keras.models.save_model(
        model,
        filepath=args.modeldir,
        overwrite=True,
        include_optimizer=False,
        save_format='tf'
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
