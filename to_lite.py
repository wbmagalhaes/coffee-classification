import tensorflow as tf

from CoffeeNet6 import create_model

model = create_model()
model.load_weights('./results/coffeenet6.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()

open('./results/coffeenet6_v0.1.tflite', 'wb').write(tfmodel)
