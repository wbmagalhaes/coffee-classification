import pytest

import os
from utils.tfrecords import load_dataset, save_tfrecords


def test_segmentation():
    # TODO: SEGMENTAÇÃO
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


def test_show_tfrecord():
    # TODO: MOSTRAR OS TFRECORDS

    assert 1 == 1


def test_train():
    # TODO: INICIAR O TREINAMENTO

    assert 1 == 1


def test_test_tfrecords():
    # TODO: TESTAR A REDE NOS TFRECORDS

    assert 1 == 1


def test_classify_images():
    # TODO: SEGMENTAR E CLASSIFICAR UMA IMAGEM

    assert 1 == 1


def test_tolite():
    # TODO: CONVERTER PARA .LITE
    assert 1 == 1


def test_to_saved_model():
    # TODO: CONVERTER PARA SAVEDMODEL
    assert 1 == 1
