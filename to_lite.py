import tensorflow as tf

from utils import reload_model

model_name = 'CoffeeNet6'
epoch = 0

model = reload_model.from_json(model_name, epoch)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

open('./results/coffeenet6_v0.1.tflite', 'wb').write(tflite_model)
