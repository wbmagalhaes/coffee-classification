import os

from utils import data_reader, tfrecords
from random import shuffle

base_dir = './images/'

img_dirs = [
    'ardido',
    'brocado',
    # 'chocho',
    # 'coco',
    # 'concha',
    'marinheiro',
    'normal',
    'preto',
    # 'quebrado',
    'verde'
]

data_dir = './data'

training_percentage = 0.8

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

data = data_reader.load(base_dir, img_dirs, cut_size=64, bg_color=(0, 0, 0))
shuffle(data)

train_num = int(len(data) * training_percentage)
teste_num = int(len(data) * (1 - training_percentage)) // 2

train_data = data[:train_num]
teste_data = data[train_num:train_num + teste_num]
valid_data = data[train_num + teste_num:]

print(f'{len(train_data)} train images.')
print(f'{len(teste_data)} teste images.')
print(f'{len(valid_data)} valid images.')


def split_data(path, data, num):
    size = len(data) // num
    for i in range(0, num + 1):
        splitted = data[size * i:size * (i + 1)]
        if len(splitted) > 0:
            tfrecords.write(f"{path}_{i}.tfrecord", splitted)


print('Writing tfrecords...')
train_path = os.path.join(data_dir, 'classification_train')
teste_path = os.path.join(data_dir, 'classification_teste')
valid_path = os.path.join(data_dir, 'classification_valid')

split_data(train_path, train_data, 1)
split_data(teste_path, teste_data, 1)
split_data(valid_path, valid_data, 1)
print('Finished.')
