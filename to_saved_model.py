import tensorflow as tf

from utils import reload_model

model_name = 'CoffeeNet6'
epoch = 0

model = reload_model.from_json(model_name, epoch)
tf.saved_model.save(model, model_name)
