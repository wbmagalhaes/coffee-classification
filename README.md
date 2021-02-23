# Coffee Beans Classification

![CoffeeNet6](classi_net.png)

Resumo

# Table of Contents

- [Coffee Beans Classification](#coffee-beans-classification)
- [Table of Contents](#table-of-contents)
- [Treinamento](#treinamento)
  - [Criar tfrecords](#criar-tfrecords)
  - [Ver tfrecords](#ver-tfrecords)
  - [Treinamento da Rede](#treinamento-da-rede)
- [Uso](#uso)
  - [Teste em tfrecords](#teste-em-tfrecords)
  - [Usar em Imagens](#usar-em-imagens)


# Treinamento
## Criar tfrecords

```
python create_tfrecords.py
```

Parâmetros Opcionais:

| **Parâmetro**   | **Padrão** | **Descrição**                                 |
| :-------------- | :--------: | :-------------------------------------------- |
| -i --inputdir   |  /images   | diretório contendo as imagens                 |
| -o --outputdir  |   /data    | diretorio onde serão criados os tfrecords     |
| --train_percent |    0.8     | porcentagem de imagens para treinamento       |
| --n_files       |   1 1 1    | quantidade de divisões nos arquivos tfrecords |

**Parâmetro -i**

Este parâmetro define o diretório onde são as imagens do dataset de grãos.

As imagens devem ser .jpg e podem estar separadas em subspastas e os arquivos .json devem estar na mesma pasta e com o mesmo nome da imagem.

**Parâmetro -o**

Este parâmetro define o diretório onde serão criados os arquivos .tfrecord usados pela rede.

Os arquivos são criados com os nomes train_dataset.tfrecord, valid_dataset.tfrecord e teste_dataset.tfrecord. Arquivos de mesmo nome serão substituídos.

**Parâmetro -train_percent**

Este parâmetro define a porcentagem de imagens que serão utilizadas no treinamento, a porcentagem restante é dividida igualmente entre validação e teste. O valor padrão de 0.8 corresponde a 80%.

**Parâmetro -n_files**

Este parâmetro define em quantos arquivos .tfrecords os dados serão divididos, isso é útil para que os arquivos não passem do limite de 100Mb do GitHub. O valor padrão de 1 1 1 corresponde a 1 arquivo para treinamento, 1 para validação e 1 para teste.

## Ver tfrecords
## Treinamento da Rede

# Uso

## Teste em tfrecords

## Usar em Imagens