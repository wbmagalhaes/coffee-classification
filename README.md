# Coffee Beans Classification

![CoffeeNet6](classi_net.png)

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
| --train_percent |    0.8     | porcentagem de imagens para treinamento             |
| --n_files       |   1 1 1    | quantidade de divisões nos arquivos TFRecord        |
| --no-shuffle    |    True    | não randomiza as imagens antes de dividir o dataset |

**Parâmetro --train_percent**

Define a porcentagem de imagens que serão utilizadas no treinamento, a porcentagem restante é dividida igualmente entre validação e teste. O valor padrão de 0.8 corresponde a 80%.

**Parâmetro --no-shuffle**

Por padrão, as imagens são randomizadas antes da divisão em treinamento, validação e teste. Se este parâmetro estiver presente, as imagens não são randomizadas.

**Parâmetro --n_files**

Define em quantos arquivos TFRecord os dados serão divididos, isso é útil para que os arquivos não passem do limite de 100Mb do GitHub. O valor padrão de 1 1 1 corresponde a 1 arquivo para treinamento, 1 para validação e 1 para teste.

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
python show_tfrecords.py -p data/teste_dataset0.tfrecord
```

Parâmetros Requeridos:

    -p --path   caminho para o arquivo .tfrecord


**Parâmetro -p**

Path

Parâmetros Opcionais:

| **Parâmetro** | **Padrão** | **Descrição**                 |
| :------------ | :--------: | :---------------------------- |
| --batch       |     36     | diretório contendo as imagens |
| --augment     |   False    | diretório contendo as imagens |

**Parâmetro --batch**

Batch

**Parâmetro --augment**

Augment

Formato do Resultado:

![Samples](dataset_samples.png)

## Treinamento da Rede

```
python train.py
```

Parâmetros Opcionais:

| **Parâmetro** | **Padrão** | **Descrição**                 |
| :------------ | :--------: | :---------------------------- |
| -i --inputdir |  /images   | diretório contendo as imagens |

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