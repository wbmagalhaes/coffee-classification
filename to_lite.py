import tensorflow as tf

from CoffeeNet6 import create_model

model = create_model()
model.load_weights('./results/coffeenet6.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

open('./results/coffeenet6_v0.1.tflite', 'wb').write(tflite_model)
