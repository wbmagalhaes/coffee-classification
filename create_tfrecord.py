import glob

from utils import config
from utils.tfrecords import write_tfrecords
from utils.data_reader import read_xml
from random import shuffle
from utils import labelmap

imgs_data = []
for _ in range(labelmap.count):
    imgs_data.append([])

addrs = glob.glob(config.IMGS_DIR + '*.xml')
images_count = len(addrs)
for i, addr in enumerate(addrs):
    filename, imgs, labels = read_xml(config.IMGS_DIR, addr)
    print(f'Lendo imagem {i + 1} de {images_count}: {filename}')

    for img, label in zip(imgs, labels):
        data = {'image': img, 'label': label}
        imgs_data[label].append(data)


def select_train_data(data_arr):
    n = len(data_arr)

    train_count = int(config.TRAIN_PERCENTAGE * n)
    val_count = int((n - train_count) / 2)

    train_arr = data_arr[:train_count]
    test_arr = data_arr[train_count:train_count + val_count]
    val_arr = data_arr[train_count + val_count:]

    return train_arr, test_arr, val_arr


train_data = []
test_data = []
val_data = []

num = min([len(imgs) for imgs in imgs_data])
num = 2500

count = 0
for imgs in imgs_data:
    count += len(imgs)

print('============')
print(f"{count} imagens carregadas.")
print('============')

for i in range(len(imgs_data)):
    print(f"{labelmap.name_of_idx(i)}: {len(imgs_data[i])}")

    shuffle(imgs_data[i])

    train_arr, test_arr, val_arr = select_train_data(imgs_data[i][:num])
    train_data.extend(train_arr)
    test_data.extend(test_arr)
    val_data.extend(val_arr)

print('============')

write_tfrecords(config.TRAINING_PATH, train_data)
print('Finished Training Data: %i Images.' % len(train_data))

write_tfrecords(config.TESTING_PATH, test_data)
print('Finished Testing Data: %i Images.' % len(test_data))

write_tfrecords(config.VALIDATION_PATH, val_data)
print('Finished Validation Data: %i Images.' % len(val_data))