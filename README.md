# Coffee Beans Classification

![Completion](https://img.shields.io/badge/completion-80%25-orange)
![Bugs](https://img.shields.io/badge/bugs-0-success)

<p align="center">
  <img src="https://raw.githubusercontent.com/wbmagalhaes/coffee-classification/master/classi_net.png" width="700">
</p>

Resumo

# Tabela de Conteúdo

- [Coffee Beans Classification](#coffee-beans-classification)
- [Tabela de Conteúdo](#tabela-de-conteúdo)
- [Treinamento](#treinamento)
  - [Segmentar Imagens](#segmentar-imagens)
  - [Criar TFRecords](#criar-tfrecords)
  - [Ver TFRecords](#ver-tfrecords)
  - [Treinamento da Rede](#treinamento-da-rede)
- [Uso](#uso)
  - [Com TFRecords](#com-tfrecords)
  - [Com Imagens](#com-imagens)

# Treinamento

Já tem uma rede treinada no diretório [models](/models) mas você pode treinar com os seus dados.

## Segmentar Imagens

TODO: Segmentar e gerar os .json

```
python segmentation.py
```

## Criar TFRecords

Para o treinamento de um novo modelo, utilize o formato [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord), no diretório [data](/data) você pode encontrar os TFRecords das imagens em [images](/images).

Você também pode criar outros TFRecords utilizando suas próprias imagens.

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

Path do TFRecord

Parâmetros Opcionais:

| **Parâmetro** | **Padrão** | **Descrição**                 |
| :------------ | :--------: | :---------------------------- |
| --batch       |     36     | diretório contendo as imagens |
| --augment     |   False    | diretório contendo as imagens |

**Parâmetro --batch**

Tamanho do Batch de imagens

**Parâmetro --augment**

Mostra as imagens após aplicar augment

Formato do Resultado:

<img src="https://raw.githubusercontent.com/wbmagalhaes/coffee-classification/master/dataset_samples.png" width="400">

Mostra o primeiro batch de imagens no arquivo TFRecords com o nome de suas respectivas classes.

## Treinamento da Rede

Como treinar a rede

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

Train path

**Parâmetro -v**

Valid path

**Parâmetro -o**

Valid path

**Parâmetro --batch**

Batch size

**Parâmetro --epochs**

Number of epochs

Parâmetros Opcionais:

| **Parâmetro**    | **Padrão** | **Descrição**                 |
| :--------------- | :--------: | :---------------------------- |
| --imsize         |     64     | diretório contendo as imagens |
| --nlayers        |     5      | diretório contendo as imagens |
| --filters        |     64     | diretório contendo as imagens |
| --kernelinit     | he_normal  | diretório contendo as imagens |
| --l2             |    0.01    | diretório contendo as imagens |
| --biasinit       |    0.1     | diretório contendo as imagens |
| --lrelualpha     |    0.02    | diretório contendo as imagens |
| --outactivation  |  softmax   | diretório contendo as imagens |
| --lr             |    1e-4    | diretório contendo as imagens |
| --labelsmoothing |    0.2     | diretório contendo as imagens |

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

# Uso

## Com TFRecords

```
python test_tfrecords.py
```

Parâmetros Opcionais:

| **Parâmetro** | **Padrão** | **Descrição**                 |
| :------------ | :--------: | :---------------------------- |
| -i --inputdir |  /images   | diretório contendo as imagens |

## Com Imagens

```
python classify_images.py
```

Parâmetros Opcionais:

| **Parâmetro** | **Padrão** | **Descrição**                 |
| :------------ | :--------: | :---------------------------- |
| -i --inputdir |  /images   | diretório contendo as imagens |