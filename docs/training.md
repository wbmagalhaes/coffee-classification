# Treinamento da Rede

Para o treinamento de um novo modelo, utilize o formato [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord), no diretório [data](data) você pode encontrar os TFRecords das imagens localizadas em [images](images).

## Segmentar Imagens

Segmentar e gerar os .json

```
python segment_images.py
```

Parâmetros:

| **Parâmetro**  | **Padrão** | **Descrição**                                 |
| :------------- | :--------: | :-------------------------------------------- |
| -i --imagesdir |   images   | diretório contendo as imagens e a segmentação |
| -o --outputdir |    None    | diretório onde serão criados os jsons         |
| --ignore       |   False    | ignora segmentação pré-existente              |
| --overwrite    |   False    | sobreescreve a segmentação pré-existente      |

## Criar TFRecords

Você pode criar TFRecords para o treinamento a partir suas próprias imagens.

```
python create_tfrecords.py
```

Parâmetros:

| **Parâmetro**   | **Padrão** | **Descrição**                                       |
| :-------------- | :--------: | :-------------------------------------------------- |
| -i --inputdir   |   images   | diretório contendo as imagens e a segmentação       |
| -o --outputdir  |    data    | diretório onde serão criados os tfrecords           |
| --im_size       |     64     | tamanho final da imagem recortada do grão           |
| --train_percent |    0.8     | porcentagem de imagens para treinamento             |
| --no-shuffle    |    True    | não randomiza as imagens antes de dividir o dataset |

**Parâmetro -i**

Define o diretório onde são as imagens do dataset de grãos.

As imagens devem estar no formato JPG, podendo estar separadas em subspastas. Os arquivos de segmentação devem estar na mesma pasta e ter o mesmo nome da imagem correspondente.

**Parâmetro -o**

Define o diretório onde serão criados os arquivos .tfrecord usados pela rede.

Os arquivos são criados com os nomes train_dataset.tfrecord, valid_dataset.tfrecord e teste_dataset.tfrecord. Arquivos de mesmo nome serão substituídos.

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
python show_tfrecords.py
```

Parâmetros:

| **Parâmetro** |         **Padrão**          | **Descrição**                   |
| :------------ | :-------------------------: | :------------------------------ |
| -p --path     | data/valid_dataset.tfrecord | caminho para o arquivo TFRecord |
| --batch       |             36              | quantidades de imagens          |
| --augment     |            False            | aplica augmentation nas imagens |

**Parâmetro -p**

Define o path para o arquivo TFRecord contendo as imagens que se deseja visualizar.

**Parâmetro --batch**

Define o número de imagens que serão mostradas. Sugiro no máximo 64 imagens.

**Parâmetro --augment**

Caso este parâmetro esteja presente, aplica o data augmentation, rotações e espelhamento nas imagens e mostra o resultado.

Formato do Resultado:

<img src="dataset_samples.png" width="500">

Mostra o primeiro batch de imagens no arquivo TFRecords com o nome de suas respectivas classes.

## Treinamento

A rede é treinada usando a biblioteca TensorFlow 2.4.1 utilizando a pipeline TFRecords.

```
python training.py
```

Parâmetros:

| **Parâmetro**    |         **Padrão**          | **Descrição**                                    |
| :--------------- | :-------------------------: | :----------------------------------------------- |
| -t --train       | data/train_dataset.tfrecord | caminho para o arquivo TFRecord de treinamento   |
| -v --valid       | data/valid_dataset.tfrecord | caminho para o arquivo TFRecord de validação     |
| --output         |      models/CoffeeNet6      | diretório onde o modelo é salvo                  |
| --logdir         |       logs/CoffeeNet6       | diretório onde os logs de treinamento são salvos |
| --batch          |             64              | tamanho do batch de imagens                      |
| --epochs         |             500             | quantidade de epochs de treinamento              |
| --im_size        |             64              | tamanho das imagens de input                     |
| --nlayers        |              5              | número de camadas de extração                    |
| --filters        |             64              | quantidade de filtros da primeira camada         |
| --kernelinit     |          he_normal          | método de inicialização dos weigths              |
| --l2             |            0.01             | valor do beta da regularização L2                |
| --biasinit       |             0.1             | valor de inicialização dos biases                |
| --lrelualpha     |            0.02             | valor do alpha da ativação LeakyReLU             |
| --outactivation  |           softmax           | ativação da última camada da rede                |
| --lr             |            1e-4             | learning rate do otimizador Adam                 |
| --labelsmoothing |             0.2             | suavização aplicada no vetor onehot              |

**Parâmetro -t**

Define o caminho para o arquivo TFRecords contendo o dataset de treinamento.

**Parâmetro -v**

Define o caminho para o arquivo TFRecords contendo o dataset de validação.

**Parâmetro -output**

Define o diretório onde será salvo o modelo.

**Parâmetro -logdir**

Define o diretório onde serão salvos os logs de treinamento para visualização no TensorBoard.

**Parâmetro --batch**

Define o tamanho da batch de imagens que será passada à rede em cada step do treinamento.

**Parâmetro --epochs**

Define o número de epochs de treinamento. A quantidade de steps por epoch é calculada automaticamente.

**Parâmetro --im_size**

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

## Export to .lite

```
python to_lite.py
```

Parâmetros:

| **Parâmetro** |    **Padrão**     | **Descrição**                              |
| :------------ | :---------------: | :----------------------------------------- |
| --modeldir    | models/CoffeeNet6 | diretório contendo o arquivo .h5 do modelo |
| --epoch       |        500        | epoch que será salva                       |
| --output      |  coffeenet6.lite  | caminho onde sera salvo o arquivo          |

## Export to Saved Model

```
python to_saved_model.py
```

Parâmetros:

| **Parâmetro** |    **Padrão**     | **Descrição**                              |
| :------------ | :---------------: | :----------------------------------------- |
| --modeldir    | models/CoffeeNet6 | diretório contendo o arquivo .h5 do modelo |
| --epoch       |        500        | epoch que será salva                       |
| --output      |    CoffeeNet6     | diretório onde sera salvo o modelo         |

- [ ] segmentation
- [x] create_tfrecords
- [x] show_tfrecords
- [ ] train
- [ ] to_saved_model
- [ ] to_lite