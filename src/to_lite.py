import tensorflow as tf

from utils import reload_model


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--modeldir', type=str, default='models/CoffeeNet6')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--output', type=int, default='coffeenet6_v1.0.tflite')

    args = parser.parse_args()

    model = reload_model.from_json(args.modeldir, args.epoch)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    open(args.output, 'wb').write(tflite_model)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
