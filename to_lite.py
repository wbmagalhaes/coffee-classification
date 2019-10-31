import tensorflow as tf

from CoffeeNet6 import create_model


def export(weights_path, out_path):
    model = create_model()
    model.load_weights(weights_path)

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
    weights_path = './results/coffeenet6.h5'
    out_path = './results/coffeenet6_v0.1.tflite'
    export(weights_path, out_path)


if __name__ == "__main__":
    main()
