import tensorflow as tf

from utils import reload_model


def convert(model_name, epoch, out_path):
    model = reload_model.from_json(model_name, epoch)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(input_details)
    print(output_details)

    open(out_path, 'wb').write(tflite_model)


def main():
    model_name = 'CoffeeNet6'
    epoch = 0
    out_path = './results/coffeenet6_v0.1.tflite'
    convert(model_name, epoch, out_path)


if __name__ == "__main__":
    main()
