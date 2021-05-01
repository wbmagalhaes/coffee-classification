import os
import pytest
import numpy as np

import tensorflow as tf

from utils.tfrecords import read_tfrecord
from utils.neural_net import load_datasets, prepare_datasets, create_model, save_model
from utils.segmentation import count_beans_pred
from utils.labelmap import label_names

from segment_images import make_segmentation, save_segmentation
from create_tfrecords import load_datafiles, save_datasets
from classify_tfrecords import classify_tfs
from classify_images import load_images, classify_imgs
from to_saved_model import export_savedmodel
from to_lite import export_tolite


def test_segmentation(tmpdir):
    data_dir = tmpdir.mkdir("data")
    json_addrs, imgs_data = make_segmentation('src/tests', False, data_dir)

    assert len(imgs_data) == 2

    assert len(imgs_data[0]) == 34 or len(imgs_data[0]) == 30
    assert len(imgs_data[1]) == 34 or len(imgs_data[1]) == 30

    save_segmentation(json_addrs, imgs_data, False)

    assert os.path.isfile(data_dir.join('20210212_164545.json'))
    assert os.path.isfile(data_dir.join('20210212_152332.json'))


def test_create_tfrecords(tmpdir):
    datasets = load_datafiles(
        input_dir='src/tests',
        im_size=64,
        train_percent=0.6,
        random=True,
        n_files=(1, 1, 1)
    )

    assert len(datasets[0]) == 15
    assert len(datasets[1]) == 5
    assert len(datasets[2]) == 5

    data_dir = tmpdir.mkdir("data")
    save_datasets(data_dir, datasets[0], datasets[1], datasets[2])

    assert os.path.isfile(data_dir.join('train_dataset.tfrecord'))
    assert os.path.isfile(data_dir.join('valid_dataset.tfrecord'))
    assert os.path.isfile(data_dir.join('teste_dataset.tfrecord'))


def test_show_tfrecord():
    dataset = read_tfrecord(['src/tests/dataset.tfrecord'])
    assert len([0 for data in dataset]) == 5


def test_train(tmpdir):
    train_ds, valid_ds = load_datasets(['src/tests/dataset.tfrecord'], ['src/tests/dataset.tfrecord'])
    train_ds, valid_ds, train_steps, valid_steps = prepare_datasets(train_ds, valid_ds, 2)

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

    model_dir = tmpdir.mkdir("result")
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
    _, _, pred = classify_tfs(
        filenames=['src/tests/dataset.tfrecord'],
        modeldir='models/saved_models/CoffeeNet6',
        im_size=64,
        batch=64
    )

    assert np.argmax(pred[0]) == 3
    assert np.argmax(pred[1]) == 3
    assert np.argmax(pred[2]) == 3


def test_classify_images():
    _, dataset = load_images(
        images_dir='src/tests',
        im_size=64,
        load_previous=False
    )

    pred = classify_imgs(
        dataset=dataset,
        modeldir='models/saved_models/CoffeeNet6'
    )

    assert len(pred) == 2

    # sometimes the images load in a diferent order
    assert pred[0].shape == (34, 6) or pred[0].shape == (30, 6)
    assert pred[1].shape == (34, 6) or pred[1].shape == (30, 6)

    counts1 = count_beans_pred(pred[0])
    counts2 = count_beans_pred(pred[1])

    expected1 = {'normal': 8, 'ardido': 0, 'brocado': 25, 'marinheiro': 0, 'preto': 0, 'verde': 1}
    expected2 = {'normal': 2, 'ardido': 6, 'brocado': 0, 'marinheiro': 22, 'preto': 0, 'verde': 0}

    def compare(prediction, expected):
        for label in label_names:
            if prediction[label] != expected[label]:
                return False

        return True

    # sometimes the images load in a diferent order
    normal_order = compare(counts1, expected1) and compare(counts2, expected2)
    invers_order = compare(counts1, expected2) and compare(counts2, expected1)
    assert normal_order or invers_order


def test_tolite(tmpdir):
    resultpath = tmpdir.mkdir("result").join('test.tflite')
    export_tolite('models/h5_models/CoffeeNet6', 500, resultpath)
    assert os.path.isfile(resultpath)


def test_to_saved_model(tmpdir):
    resultpath = tmpdir.mkdir("result")
    export_savedmodel('models/h5_models/CoffeeNet6', 500, resultpath)
    model = tf.keras.models.load_model(resultpath)
    assert model != None
