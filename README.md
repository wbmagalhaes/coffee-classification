# Classificação de grãos de café arábica

![Last](https://img.shields.io/github/last-commit/wbmagalhaes/coffee-classification)
![Completion](https://img.shields.io/badge/completion-80%25-orange)

![Python](https://img.shields.io/github/pipenv/locked/python-version/wbmagalhaes/coffee-classification)
![Tensorflow](https://img.shields.io/github/pipenv/locked/dependency-version/wbmagalhaes/coffee-classification/tensorflow)
[![Tests](https://github.com/wbmagalhaes/coffee-classification/actions/workflows/python-package.yml/badge.svg)](https://github.com/wbmagalhaes/coffee-classification/actions)
![Issues](https://img.shields.io/github/issues/wbmagalhaes/coffee-classification)

<p align="center">
  <img src="https://raw.githubusercontent.com/wbmagalhaes/coffee-classification/main/docs/classi_net.png" width="600">
</p>

Rede neural para classificação de defeitos em grãos crus de café arábica. Desenvolvida por William Bernardes Magalhães como parte do projeto de Mestrado para obtenção do título de Mestre em Química pela Universidade Estadual de Londrina.

# Tabela de Conteúdos

- [Classificação de grãos de café arábica](#classificação-de-grãos-de-café-arábica)
- [Tabela de Conteúdos](#tabela-de-conteúdos)
- [Requisitos](#requisitos)
- [Uso](#uso)
  - [Com TFRecords](#com-tfrecords)
  - [Com Imagens](#com-imagens)
- [Cite Este Projeto](#cite-este-projeto)


# Requisitos

- Python 3.8
- Tensorflow 2.4.1

# Uso

Uma rede de exemplo treinada com imagens de grãos de café arábica pode ser encontrada no diretório [models](models), mas você pode treinar uma nova rede utilizando seus dados.

A documentação para o treinamento de uma rede pode ser encontrada em [Treinamento](docs/training.md)

## Com TFRecords

Pode classificar o tfrecord de teste para avaliar a rede

Mostra a matriz de confusão

```
python classify_tfrecords.py
```

Parâmetros:

| **Parâmetro** | **Padrão** | **Descrição**                 |
| :------------ | :--------: | :---------------------------- |
| -i --inputdir |  /images   | diretório contendo as imagens |
| -m --modeldir |  /images   | diretório contendo as imagens |
| --batch       |  /images   | diretório contendo as imagens |

## Com Imagens

Pode classificar qualquer imagem

Se tiver segmentada usa o .json

Se tiver que segmentar, usa o .jpg

```
python classify_images.py
```

Parâmetros:

| **Parâmetro** | **Padrão** | **Descrição**                 |
| :------------ | :--------: | :---------------------------- |
| -i --inputdir |  /images   | diretório contendo as imagens |

# Cite Este Projeto

```
@misc{asd,
  author =       {William Bernardes Magalhães},
  title =        {TITLE},
  howpublished = {\url{https://github.com/wbmagalhaes/coffee-classification}},
  year =         {2021}
}
```

Implemented

- [x] segmentation
- [x] create_tfrecords
- [x] show_tfrecords
- [x] train
- [x] classify_tfrecords
- [x] classify_images
- [x] to_saved_model
- [x] to_lite

PyTest

- [x] segmentation
- [x] create_tfrecords
- [x] show_tfrecords
- [x] train
- [x] classify_tfrecords
- [x] classify_images
- [x] to_saved_model
- [x] to_lite

README

- [ ] segmentation
- [x] create_tfrecords
- [x] show_tfrecords
- [ ] train
- [ ] classify_tfrecords
- [ ] classify_images
- [ ] to_saved_model
- [ ] to_lite
