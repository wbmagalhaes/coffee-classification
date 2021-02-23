# Coffee Beans Classification

![CoffeeNet6](classi_net.png)

## Criar os TFRecords

create_tfrecords.py
    --inputdir -i   diretório onde estão as imagems. Padrão "/images"
    --outputdir -o  diretorio onde serão criados os tfrecords. Padrão: "/data"
    --train_percent porcentagem de imagens para treinamento. Padrão: 0.8 (80%)
    --splits        quantidade de divisões nos arquivos tfrecords (para não passar do limite de tamanho do github). Padrão: 1 1 1 (sem divisões)
