import os
import pytest

from utils.tfrecords import load_datafiles, save_tfrecords
from utils.tfrecords import read_tfrecord
from utils.CoffeeNet import load_datasets, create_model, save_model


def test_segmentation():
    # TODO: SEGMENTAÇÃO

    assert 1 == 1


def test_create_tfrecord(tmpdir):

    dataset = load_datafiles(
        input_dir='src/tests',
        im_size=64,
        training_percentage=0.6,
        random=True,
        n_files=(1, 1, 1)
    )

    assert len(dataset[0]) == 15
    assert len(dataset[1]) == 5
    assert len(dataset[2]) == 5

    data_dir = tmpdir.mkdir("data")

    save_tfrecords(dataset[0], 'train_dataset', data_dir, n=1)
    assert os.path.isfile(data_dir.join('train_dataset.tfrecord'))

    save_tfrecords(dataset[1], 'valid_dataset', data_dir, n=1)
    assert os.path.isfile(data_dir.join('valid_dataset.tfrecord'))

    save_tfrecords(dataset[2], 'teste_dataset', data_dir, n=1)
    assert os.path.isfile(data_dir.join('teste_dataset.tfrecord'))


def test_show_tfrecord():
    dataset = read_tfrecord(['src/tests/dataset.tfrecord'])
    assert len([0 for data in dataset]) == 3


def test_train(tmpdir):
    train_ds, valid_ds, train_steps, valid_steps = load_datasets(
        ['src/tests/dataset.tfrecord'],
        ['src/tests/dataset.tfrecord'],
        2
    )

    assert train_steps == 2
    assert valid_steps == 2

    model = create_model(
        input_shape=(64, 64, 3),
        num_layers=1,
        filters=16,
        kernel_initializer='he_normal',
        l2=0.01,
        bias_value=0,
        leaky_relu_alpha=0.02,
        output_activation='softmax',
        lr=1e-4,
        label_smoothing=0
    )

    model_dir = tmpdir.mkdir("models")
    save_model(model, model_dir)

    assert os.path.isfile(model_dir.join('model.json'))

    history = model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        epochs=1,
        verbose=1,
        validation_data=valid_ds,
        validation_freq=1,
        validation_steps=valid_steps
    )

    acc = history.history['logits_categorical_accuracy']
    assert len(acc) == 1


def test_classify_tfrecords():
    dataset = tfrecords.read_tfrecord(['src/tests/dataset.tfrecord'])
    x_data, y_true = zip(*[data for data in dataset])

    model = reload_model.from_savedmodel('models/CoffeeNet6')
    _, y_pred = model.predict(dataset.batch(64))

    assert np.argmax(y_pred[0]) == 3
    assert np.argmax(y_pred[1]) == 3
    assert np.argmax(y_pred[2]) == 3


def test_classify_images():
    # TODO: SEGMENTAR E CLASSIFICAR UMA IMAGEM

    assert 1 == 1


# def test_tolite(tmpdir):
#     model = reload_model.from_json('models/CoffeeNet6', 500)

#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     tflite_model = converter.convert()

#     interpreter = tf.lite.Interpreter(model_content=tflite_model)
#     interpreter.allocate_tensors()

#     resultpath = tmpdir.mkdir("result").join('test.tflite')

#     open(resultpath, 'wb').write(tflite_model)

#     assert os.path.isfile(resultpath)


# def test_to_saved_model(tmpdir):
#     model = reload_model.from_json('models/CoffeeNet6', 500)

#     resultpath = tmpdir.mkdir("result")
#     tf.keras.models.save_model(
#         model,
#         filepath=resultpath,
#         overwrite=True,
#         include_optimizer=False,
#         save_format='tf'
#     )

#     model = reload_model.from_savedmodel(resultpath)
#     assert model != None
