import tensorflow as tf

from utils import reload_model


def convert(model_name, epoch):
    model = reload_model.from_json(model_name, epoch)
    tf.saved_model.save(model, model_name)


def main():
    model_name = 'CoffeeNet6'
    epoch = 0
    convert(model_name, epoch)


if __name__ == "__main__":
    main()
