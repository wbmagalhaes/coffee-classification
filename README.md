# Coffee Beans Classification

![CoffeeNet6](classi_net.png)

## Criar os TFRecords

create_tfrecords.py

**Parameter** | **Padrão** | **Descrição**
------------- | ---------- | -------------
-i --inputdir | /images | diretório contendo as imagems
-o --outputdir | /data | diretorio onde serão criados os tfrecords
--train_percent | 0.8 (80%) | porcentagem de imagens para treinamento
--splits | 1 1 1 (sem divisões) | quantidade de divisões nos arquivos tfrecords (para não passar do limite de tamanho do github)
