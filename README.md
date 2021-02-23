# Coffee Beans Classification

![CoffeeNet6](classi_net.png)

Resumo

# Tabela de Conteúdo

- [Coffee Beans Classification](#coffee-beans-classification)
- [Tabela de Conteúdo](#tabela-de-conteúdo)
- [Treinamento](#treinamento)
  - [Criar tfrecords](#criar-tfrecords)
  - [Ver tfrecords](#ver-tfrecords)
  - [Treinamento da Rede](#treinamento-da-rede)
- [Uso](#uso)
  - [Com tfrecords](#com-tfrecords)
  - [Com Imagens](#com-imagens)

# Treinamento

Já tem uma rede treinada em `/models` mas você pode treinar com os seus dados.

## Criar tfrecords

Para treinar, utilize os tfrecords do tensorflow, já tem na pasta `/data` mas você pode criar rodando o arquivo.

```
python create_tfrecords.py -i <images path> -o <tfrecords path>
```

**Exemplo:**

```
python create_tfrecords.py -i /images -o /data
```

**Parâmetros Requeridos:**

    -i --inputdir   diretório contendo as imagens
    -o --outputdir  diretório onde serão criados os tfrecords


**Parâmetro -i**

Este parâmetro define o diretório onde são as imagens do dataset de grãos.

As imagens devem ser .jpg e podem estar separadas em subspastas e os arquivos .json devem estar na mesma pasta e com o mesmo nome da imagem.

**Parâmetro -o**

Este parâmetro define o diretório onde serão criados os arquivos .tfrecord usados pela rede.

Os arquivos são criados com os nomes train_dataset.tfrecord, valid_dataset.tfrecord e teste_dataset.tfrecord. Arquivos de mesmo nome serão substituídos.

**Parâmetros Opcionais:**

| **Parâmetro**   | **Padrão** | **Descrição**                                       |
| :-------------- | :--------: | :-------------------------------------------------- |
| --train_percent |    0.8     | porcentagem de imagens para treinamento             |
| --no-shuffle    |    True    | não randomiza as imagens antes de dividir o dataset |
| --n_files       |   1 1 1    | quantidade de divisões nos arquivos tfrecords       |

**Parâmetro --train_percent**

Este parâmetro define a porcentagem de imagens que serão utilizadas no treinamento, a porcentagem restante é dividida igualmente entre validação e teste. O valor padrão de 0.8 corresponde a 80%.

**Parâmetro --no-shuffle**

Por padrão, as imagens são randomizadas antes da divisão em treinamento, validação e teste. Se este parâmetro estiver presente, as imagens não são randomizadas.

**Parâmetro --n_files**

Este parâmetro define em quantos arquivos .tfrecords os dados serão divididos, isso é útil para que os arquivos não passem do limite de 100Mb do GitHub. O valor padrão de 1 1 1 corresponde a 1 arquivo para treinamento, 1 para validação e 1 para teste.

## Ver tfrecords
```
python show_tfrecords.py
```

Parâmetros Opcionais:

| **Parâmetro** | **Padrão** | **Descrição**                 |
| :------------ | :--------: | :---------------------------- |
| -i --inputdir |  /images   | diretório contendo as imagens |

## Treinamento da Rede
```
python train.py
```

Parâmetros Opcionais:

| **Parâmetro** | **Padrão** | **Descrição**                 |
| :------------ | :--------: | :---------------------------- |
| -i --inputdir |  /images   | diretório contendo as imagens |

# Uso

## Com tfrecords
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