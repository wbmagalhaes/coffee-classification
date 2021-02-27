# Classificação de grãos de café arábica

![Python](https://img.shields.io/github/pipenv/locked/python-version/wbmagalhaes/coffee-classification)
![Tensorflow](https://img.shields.io/github/pipenv/locked/dependency-version/wbmagalhaes/coffee-classification/tensorflow)
[![Tests](https://github.com/wbmagalhaes/coffee-classification/actions/workflows/python-package.yml/badge.svg)](https://github.com/wbmagalhaes/coffee-classification/actions)
![Issues](https://img.shields.io/github/issues/wbmagalhaes/coffee-classification)

<p align="center">
  Rede neural para classificação de defeitos em grãos crus de café arábica.
  <img src="docs/classi_net.png" width="600">
</p>

Desenvolvida por William Bernardes Magalhães como parte do projeto de Mestrado para obtenção do título de Mestre em Química pela Universidade Estadual de Londrina.

# Tabela de Conteúdos

- [Classificação de grãos de café arábica](#classificação-de-grãos-de-café-arábica)
- [Tabela de Conteúdos](#tabela-de-conteúdos)
- [Requisitos](#requisitos)
- [Uso](#uso)
  - [Classificar TFRecords](#classificar-tfrecords)
  - [Classificar Imagens](#classificar-imagens)
- [Cite Este Projeto](#cite-este-projeto)

# Requisitos

- Python 3.8
- Tensorflow 2.4.1

# Uso

Existem duas formas de realizar a classificação, utilizando o formato [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) ou utilizando imagens no formato JPG.

Para utilizar o TFRecord, as imagens já devem ter sido segmentadas e os grãos devem ter classificação conhecida. O TFRecord do dataset de testes pode ser utilizado como exemplo. Detalhes de como gerar TFRecords a partir de seus dados podem ser encontrados em [Criar TFRecords](docs/training.md#criar-tfrecords).

Para classificar imagens diretamente blablabla

Uma rede de exemplo treinada com imagens de grãos de café arábica pode ser encontrada no diretório [models](models), mas você pode treinar uma nova rede utilizando seus dados. A documentação para o treinamento de uma rede pode ser encontrada em [Treinamento](docs/training.md#treinamento).

## Classificar TFRecords

Utilizando o TFRecord, os grãos já tem classificação conhecida.

Ao final, é mostrado a matriz de confusão, comparando a classificação esperada e a classificação obtida.

```
python classify_tfrecords.py
```

Parâmetros:

| **Parâmetro** |         **Padrão**          | **Descrição**                |
| :------------ | :-------------------------: | :--------------------------- |
| -i --inputdir | data/teste_dataset.tfrecord | caminho até o TFRecords      |
| -m --modeldir |      models/CoffeeNet6      | diretório contendo o modelo  |
| --im_size     |             64              | tamanho das imagens de input |
| --batch       |             36              | número de imagens por batch  |

## Classificar Imagens

Pode classificar qualquer imagem

Se tiver segmentada usa o .json

Se tiver que segmentar, usa o .jpg

```
python classify_images.py
```

Parâmetros:

| **Parâmetro** |         **Padrão**          | **Descrição**                    |
| :------------ | :-------------------------: | :------------------------------- |
| -i --inputdir | data/teste_dataset.tfrecord | caminho até o TFRecords          |
| -m --modeldir |      models/CoffeeNet6      | diretório contendo o modelo      |
| --im_size     |             64              | tamanho das imagens de input     |
| --ignore      |            False            | ignora segmentação pré-existente |

# Cite Este Projeto

```
@misc{magalhaes2021,
  author =  {William Bernardes Magalh{\~a}es},
  title  =  {Classificação de defeitos em grãos de café arábica},
  url    =  {https://github.com/wbmagalhaes/coffee-classification},
  year   =  {2021}
}
```

- [ ] classify_tfrecords
- [ ] classify_images
