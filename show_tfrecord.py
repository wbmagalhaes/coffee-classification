from utils import tfrecords, visualize
from utils.augmentation import color, zoom, rotate, flip, gaussian, clip01

ds = tfrecords.read(['./data/classification_test.tfrecord'], num_labels=6)#.shuffle(buffer_size=100)

visualize.plot_dataset(ds.batch(36))

# ds = color(
#     ds,
#     hue=0.05,
#     saturation=(0.9, 1.05),
#     brightness=0.1,
#     contrast=(0.9, 1.05))
# ds = zoom(ds, im_size=64)
# ds = gaussian(ds, stddev=0.01)
ds = rotate(ds)
ds = flip(ds)
ds = clip01(ds)

batch1 = ds.batch(36)
batch2 = ds.batch(36)
batch3 = ds.batch(36)
batch4 = ds.batch(36)
batch5 = ds.batch(36)
batch6 = ds.batch(36)
batch7 = ds.batch(36)
batch8 = ds.batch(36)

visualize.plot_dataset(batch1)
visualize.plot_dataset(batch2)
visualize.plot_dataset(batch3)
visualize.plot_dataset(batch4)
visualize.plot_dataset(batch5)
visualize.plot_dataset(batch6)
visualize.plot_dataset(batch7)
visualize.plot_dataset(batch8)
