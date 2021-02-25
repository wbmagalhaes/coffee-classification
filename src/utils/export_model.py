import tensorflow as tf
from utils.reload_model import from_json


def export_tolite(modeldir, epoch, output):
    model = from_json(modeldir, epoch)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    open(output, 'wb').write(tflite_model)


def export_savedmodel(modeldir, epoch, output):
    model = from_json(modeldir, epoch)

    tf.keras.models.save_model(
        model,
        filepath=output,
        overwrite=True,
        include_optimizer=False,
        save_format='tf'
    )
