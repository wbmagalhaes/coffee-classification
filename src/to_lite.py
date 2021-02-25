import sys
import argparse

import tensorflow as tf
from utils.reload_model import from_json


def export_tolite(modeldir, epoch, output):
    model = from_json(modeldir, epoch)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    open(output, 'wb').write(tflite_model)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', type=str, default='models/CoffeeNet6')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--output', type=int, default='coffeenet6_v1.0.tflite')
    args = parser.parse_args()

    export_tolite(args.modeldir, args.epoch, args.output)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
