import glob

from utils import config
from utils.tfrecords import write_tfrecords
from utils.data_reader import read_xml
from random import shuffle
from collections import defaultdict
from utils import labelmap

imgs_data = []
label_counter = defaultdict(int)
for addr in glob.glob(config.IMGS_DIR + '*.xml'):
    _, imgs, labels = read_xml(config.IMGS_DIR, addr)

    for img, label in zip(imgs, labels):
        data = {'image': img, 'label': label}
        imgs_data.append(data)

        label_counter[label] += 1

print('============')
for l in label_counter:
    print(labelmap.name_of_idx(l),':', label_counter[l])
print('============')

shuffle(imgs_data)
img_count = len(imgs_data)
print(img_count, 'Images loaded.')

train_count = int(config.TRAIN_PERCENTAGE * img_count)
val_count = int((img_count - train_count) / 2)

train_data = imgs_data[0:train_count]
test_data = imgs_data[train_count:train_count + val_count]
val_data = imgs_data[train_count + val_count:]

write_tfrecords(config.TRAINING_PATH, train_data)
print('Finished Training Data: %i Images.' % len(train_data))

write_tfrecords(config.TESTING_PATH, test_data)
print('Finished Testing Data: %i Images.' % len(test_data))

write_tfrecords(config.VALIDATION_PATH, val_data)
print('Finished Validation Data: %i Images.' % len(val_data))
