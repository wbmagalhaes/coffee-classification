import pytest

import os
from utils.tfrecords import load_dataset, save_tfrecords


def test_classify_images():
    assert 1 == 1


def test_create_tfrecord():

    dataset = load_dataset(
        input_dir='tests/images',
        im_size=64,
        training_percentage=0.6,
        random=True,
        n_files=(1, 1, 1)
    )

    assert len(dataset[0]) == 15
    assert len(dataset[1]) == 5
    assert len(dataset[2]) == 5

    save_tfrecords(dataset[0], 'train_dataset', 'tests/data', n=1)
    assert os.path.isfile('tests/data/train_dataset.tfrecord')

    save_tfrecords(dataset[1], 'valid_dataset', 'tests/data', n=1)
    assert os.path.isfile('tests/data/valid_dataset.tfrecord')

    save_tfrecords(dataset[2], 'teste_dataset', 'tests/data', n=1)
    assert os.path.isfile('tests/data/teste_dataset.tfrecord')


def test_segmentation():
    assert 1 == 1


def test_show_tfrecord():
    assert 1 == 1


def test_test_tfrecords():
    assert 1 == 1


def test_tolite():
    assert 1 == 1


def test_to_saved_model():
    assert 1 == 1


def test_train():
    assert 1 == 1
