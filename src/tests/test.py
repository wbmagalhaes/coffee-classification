import os
import pytest
import numpy as np

from utils.tfrecords import read_tfrecord
from utils.CoffeeNet import load_datasets, create_model, save_model

from segmentation import segment_images, save_segmentation

from create_tfrecords import create, save
from classify_tfrecords import classify

from to_saved_model import export_savedmodel
from to_lite import export_tolite


def test_segmentation(tmpdir):
    imgs_data = segment_images('src/tests')

    assert len(imgs_data) == 2

    assert imgs_data['20210212_164545.json']['data'] == None
    assert len(imgs_data['20210212_152332.json']['data']) == 34

    data_dir = tmpdir.mkdir("data")
    save_segmentation(imgs_data, data_dir)

    assert not os.path.isfile(data_dir.join('20210212_164545.json'))
    assert os.path.isfile(data_dir.join('20210212_152332.json'))


def test_create_tfrecords(tmpdir):
    dataset = create(
        input_dir='src/tests',
        im_size=64,
        train_percent=0.6,
        random=True,
        n_files=(1, 1, 1)
    )

    assert len(dataset[0]) == 15
    assert len(dataset[1]) == 5
    assert len(dataset[2]) == 5

    data_dir = tmpdir.mkdir("data")
    save(data_dir, dataset[0], dataset[1], dataset[2])

    assert os.path.isfile(data_dir.join('train_dataset.tfrecord'))
    assert os.path.isfile(data_dir.join('valid_dataset.tfrecord'))
    assert os.path.isfile(data_dir.join('teste_dataset.tfrecord'))


def test_show_tfrecord():
    dataset = read_tfrecord(['src/tests/dataset.tfrecord'])
    assert len([0 for data in dataset]) == 5


def test_train(tmpdir):
    train_ds, valid_ds, train_steps, valid_steps = load_datasets(
        ['src/tests/dataset.tfrecord'],
        ['src/tests/dataset.tfrecord'],
        2
    )

    assert train_steps == 3
    assert valid_steps == 3

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
    _, _, pred = classify(
        filenames=['src/tests/dataset.tfrecord'],
        modeldir='models/CoffeeNet6',
        batch=64
    )

    assert np.argmax(pred[0]) == 3
    assert np.argmax(pred[1]) == 3
    assert np.argmax(pred[2]) == 3


def test_classify_images():
    # TODO: SEGMENTAR E CLASSIFICAR UMA IMAGEM

    assert 1 == 1


# def test_tolite(tmpdir):
#     resultpath = tmpdir.mkdir("result").join('test.tflite')
#     export_tolite('models/CoffeeNet6', 500, resultpath)
#     assert os.path.isfile(resultpath)


# def test_to_saved_model(tmpdir):
#     resultpath = tmpdir.mkdir("result")
#     export_savedmodel('models/CoffeeNet6', 500, resultpath)
#     model = tf.keras.models.load_model(resultpath)
#     assert model != None
