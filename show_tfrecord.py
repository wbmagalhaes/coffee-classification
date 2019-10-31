import tfrecords
import augmentation
import utils

dataset = tfrecords.read(['./data/data_train.tfrecord']).shuffle(buffer_size=10000)
dataset = dataset.map(utils.normalize, num_parallel_calls=4)
utils.plot_dataset(dataset.batch(64))

dataset = augmentation.apply(dataset)
utils.plot_dataset(dataset.batch(64))
