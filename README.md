# Classificação de grãos de café arábica

![Last](https://img.shields.io/github/last-commit/wbmagalhaes/coffee-classification)
![Completion](https://img.shields.io/badge/completion-80%25-orange)

[![Tests](https://github.com/wbmagalhaes/coffee-classification/actions/workflows/python-package.yml/badge.svg)](https://github.com/wbmagalhaes/coffee-classification/actions)
![Issues](https://img.shields.io/github/issues/wbmagalhaes/coffee-classification)
![Python](https://img.shields.io/github/pipenv/locked/python-version/wbmagalhaes/coffee-classification)
![Tensorflow](https://img.shields.io/github/pipenv/locked/dependency-version/wbmagalhaes/coffee-classification/tensorflow)

<p align="center">
  <img src="https://raw.githubusercontent.com/wbmagalhaes/coffee-classification/master/classi_net.png" width="600">
</p>

Rede neural desenvolvida por William Bernardes Magalhães como parte do projeto de Mestrado para obtenção do título de Mestre em Química pela Universidade Estadual de Londrina.

# Tabela de Conteúdo

- [Classificação de grãos de café arábica](#classificação-de-grãos-de-café-arábica)
- [Tabela de Conteúdo](#tabela-de-conteúdo)
- [Requisitos](#requisitos)
- [Treinamento](#treinamento)
  - [Segmentar Imagens](#segmentar-imagens)
  - [Criar TFRecords](#criar-tfrecords)
  - [Ver TFRecords](#ver-tfrecords)
  - [Treinamento da Rede](#treinamento-da-rede)
- [Uso](#uso)
  - [Com TFRecords](#com-tfrecords)
  - [Com Imagens](#com-imagens)
- [Cite Este Projeto](#cite-este-projeto)


# Requisitos

- Python 3.8
- Tensorflow 2.4.1

# Treinamento

Já existe uma rede de exemplo treinada com imagens de grãos de café arábica no diretório [models](/models), mas você pode treinar uma nova rede utilizando seus dados.

Para o treinamento de um novo modelo, utilize o formato [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord), no diretório [data](/data) você pode encontrar os TFRecords das imagens localizadas em [images](/images).

## Segmentar Imagens

TODO: Segmentar e gerar os .json

```
python segmentation.py
```

## Criar TFRecords

Você pode criar TFRecords para o treinamento a partir suas próprias imagens.

```
python create_tfrecords.py -i <images path> -o <tfrecords path>
```

Exemplo:

```
python create_tfrecords.py -i images -o data
```

Parâmetros Requeridos:

    -i --inputdir   diretório contendo as imagens e a segmentação
    -o --outputdir  diretório onde serão criados os tfrecords

**Parâmetro -i**

Define o diretório onde são as imagens do dataset de grãos.

As imagens devem estar no formato JPG, podendo estar separadas em subspastas. Os arquivos de segmentação devem estar na mesma pasta e ter o mesmo nome da imagem correspondente.

**Parâmetro -o**

Define o diretório onde serão criados os arquivos .tfrecord usados pela rede.

Os arquivos são criados com os nomes train_dataset.tfrecord, valid_dataset.tfrecord e teste_dataset.tfrecord. Arquivos de mesmo nome serão substituídos.

Parâmetros Opcionais:

| **Parâmetro**   | **Padrão** | **Descrição**                                       |
| :-------------- | :--------: | :-------------------------------------------------- |
| --im_size       |     64     | tamanho final da imagem recortada do grão           |
| --train_percent |    0.8     | porcentagem de imagens para treinamento             |
| --no-shuffle    |    True    | não randomiza as imagens antes de dividir o dataset |

**Parâmetro --im_size**

Define o redimensionamento do recorte quadrado do grão de café.

**Parâmetro --train_percent**

Define a porcentagem de imagens que serão utilizadas no treinamento, a porcentagem restante é dividida igualmente entre validação e teste. O valor padrão de 0.8 corresponde a 80%.

**Parâmetro --no-shuffle**

Por padrão, as imagens são randomizadas antes da divisão em treinamento, validação e teste. Se este parâmetro estiver presente, as imagens não são randomizadas.

Formato do Resultado:

```
4275 total images
normal: 1149
ardido: 1139
brocado: 404
marinheiro: 307
preto: 615
verde: 661

3420 train images
normal: 912
ardido: 910
brocado: 334
marinheiro: 241
preto: 500
verde: 523

427 valid images
normal: 114
ardido: 114
brocado: 39
marinheiro: 31
preto: 53
verde: 76

428 teste images
normal: 123
ardido: 115
brocado: 31
marinheiro: 35
preto: 62
verde: 62
```

Após carregar as imagens e gerar os TFRecords, será mostrado uma lista contendo a quantidade de imagens de cada classe separadas nos datasets de treinamento, validação e teste.

## Ver TFRecords

Para verificar as imagens armazenadas nos TFRecords.

```
python show_tfrecords.py -p <tfrecords path>
```

Exemplo:

```
python show_tfrecords.py -p data/teste_dataset.tfrecord
```

Parâmetros Requeridos:

    -p --path   caminho para o arquivo TFRecord

**Parâmetro -p**

Define o path para o arquivo TFRecord contendo as imagens que se deseja visualizar.

Parâmetros Opcionais:

| **Parâmetro** | **Padrão** | **Descrição**                   |
| :------------ | :--------: | :------------------------------ |
| --batch       |     36     | quantidades de imagens          |
| --augment     |   False    | aplica augmentation nas imagens |

**Parâmetro --batch**

Define o número de imagens que serão mostradas. Sugiro no máximo 64 imagens.

**Parâmetro --augment**

Caso este parâmetro esteja presente, aplica o data augmentation, rotações e espelhamento nas imagens e mostra o resultado.

Formato do Resultado:

<img src="https://raw.githubusercontent.com/wbmagalhaes/coffee-classification/master/dataset_samples.png" width="500">

Mostra o primeiro batch de imagens no arquivo TFRecords com o nome de suas respectivas classes.

## Treinamento da Rede

A rede é treinada usando a biblioteca TensorFlow 2.4.1 utilizando a pipeline TFRecords.

```
python train.py -t <train file path> -v <validation file path> -o <output directory> --batch <batch size> --epochs <epochs number>
```

Exemplo:

```
python train.py -t data/train_dataset.tfrecord -v data/valid_dataset.tfrecord -o CoffeeNet6 --batch 64 --epochs 500
```

Parâmetros Requeridos:

    -t --train    caminho para o arquivo TFRecord de treinamento
    -v --valid    caminho para o arquivo TFRecord de validação
    -o --output   diretório onde o modelo e os logs de treinamento serão salvos
    --batch       tamanho do batch de imagens
    --epochs      quantidade de epochs de treinamento

**Parâmetro -t**

Define o caminho para o arquivo TFRecords contendo o dataset de treinamento.

**Parâmetro -v**

Define o caminho para o arquivo TFRecords contendo o dataset de validação.

**Parâmetro -o**

Define o diretório onde serão salvos o modelo e os logs de treinamento para visualização no TensorBoard.

**Parâmetro --batch**

Define o tamanho da batch de imagens que será passada à rede em cada step do treinamento.

**Parâmetro --epochs**

Define o número de epochs de treinamento. A quantidade de steps por epoch é calculada automaticamente.

Parâmetros Opcionais:

| **Parâmetro**    | **Padrão** | **Descrição**                            |
| :--------------- | :--------: | :--------------------------------------- |
| --imsize         |     64     | tamanho das imagens de input             |
| --nlayers        |     5      | número de camadas de extração            |
| --filters        |     64     | quantidade de filtros da primeira camada |
| --kernelinit     | he_normal  | método de inicialização dos weigths      |
| --l2             |    0.01    | valor do beta da regularização L2        |
| --biasinit       |    0.1     | valor de inicialização dos biases        |
| --lrelualpha     |    0.02    | valor do alpha da ativação LeakyReLU     |
| --outactivation  |  softmax   | ativação da última camada da rede        |
| --lr             |    1e-4    | learning rate do otimizador Adam         |
| --labelsmoothing |    0.2     | suavização aplicada no vetor onehot      |

**Parâmetro --imsize**

Img size

**Parâmetro --nlayers**

Number of layers

**Parâmetro --filters**

Number of filters in the first layer

**Parâmetro --kernelinit**

Weights initialization

**Parâmetro --l2**

L2 regularization

**Parâmetro --biasinit**

Biases initialization

**Parâmetro --lrelualpha**

LeakyReLU alpha

**Parâmetro --outactivation**

Output activation

**Parâmetro --lr**

Learning rate

**Parâmetro --labelsmoothing**

Label smoothing

Formato do Resultado:

```
Keras printa o modelo
Keras printa o treinamento
Tensorboard mostra o gráfico
```

# Uso

Após treinada, você pode usar.

## Com TFRecords

Pode classificar o tfrecord de teste para avaliar a rede

Mostra a matriz de confusão

```
python test_tfrecords.py
```

Parâmetros Opcionais:

| **Parâmetro** | **Padrão** | **Descrição**                 |
| :------------ | :--------: | :---------------------------- |
| -i --inputdir |  /images   | diretório contendo as imagens |

## Com Imagens

Pode classificar qualquer imagem

Se tiver segmentada usa o .json

Se tiver que segmentar, usa o .jpg

```
python classify_images.py
```

Parâmetros Opcionais:

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
