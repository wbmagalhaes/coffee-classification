from utils import tfrecords, visualize
from utils.augmentation import color, zoom, rotate, flip, gaussian, clip01

ds = tfrecords.read(['./data/classification_teste_0.tfrecord'], num_labels=6)#.shuffle(buffer_size=100)

visualize.plot_dataset(ds.batch(36))

ds = rotate(ds)
ds = flip(ds)
ds = clip01(ds)

visualize.plot_dataset(ds.batch(36))
