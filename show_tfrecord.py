from utils import tfrecords, visualize
from utils.augmentation import color, zoom, rotate, flip, gaussian, clip01

ds = tfrecords.read(['./data/classification_test.tfrecord'], num_labels=6).shuffle(buffer_size=100)
visualize.plot_dataset(ds.batch(64))

ds = color(
    ds,
    hue=0.05,
    saturation=(0.9, 1.05),
    brightness=0.1,
    contrast=(0.9, 1.05))
ds = zoom(ds, im_size=64)
ds = gaussian(ds, stddev=0.01)
ds = rotate(ds)
ds = flip(ds)
ds = clip01(ds)

visualize.plot_dataset(ds.batch(64))
