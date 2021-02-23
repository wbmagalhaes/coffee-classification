# Coffee Beans Classification

![CoffeeNet6](classi_net.png)

## Criar os TFRecords

create_tfrecords.py

| **Parametro**   | **Padrão** | **Descrição**                                 |
| :-------------- | :--------: | :-------------------------------------------- |
| -i --inputdir   |  /images   | diretório contendo as imagens                 |
| -o --outputdir  |   /data    | diretorio onde serão criados os tfrecords     |
| --train_percent |    0.8     | porcentagem de imagens para treinamento       |
| --splits        |   1 1 1    | quantidade de divisões nos arquivos tfrecords |

**-i** 

Este parâmetro define o diretório onde são as imagens do dataset de grãos.
Os arquivos .json devem estar na mesma pasta e com o mesmo nome da imagem.
As imagens buscadas devem ser .jpg e podem estar separadas em subspastas.

**-o** 

Este parâmetro define o diretório onde serão criados os arquivos .tfrecord usados pela rede.
Os arquivos são criados com os nomes  *train_dataset.tfrecord*, *valid_dataset.tfrecord* e *teste_dataset.tfrecord*. Arquivos de mesmo nome serão substituídos.


