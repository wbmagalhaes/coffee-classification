import tensorflow as tf
import json


def from_json(path, epoch):
    # Load model
    with open(f'{path}/model.json', 'r') as f:
        json_config = json.load(f)

    model = tf.keras.models.model_from_json(json_config)

    # Recover weights
    model.load_weights(f'{path}/epoch-{epoch:04d}.h5')
    return model


def from_savedmodel(path):
    return tf.keras.models.load_model(path)
