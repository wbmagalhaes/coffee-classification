from utils import tfrecords, augmentation, other, visualize

dataset = tfrecords.read(['./data/data_train.tfrecord']).shuffle(buffer_size=10000)
dataset = dataset.map(other.normalize, num_parallel_calls=4)
visualize.plot_dataset(dataset.batch(64))

dataset = augmentation.apply(dataset)
visualize.plot_dataset(dataset.batch(64))
