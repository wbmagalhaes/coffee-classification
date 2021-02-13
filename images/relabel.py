import os
import json
import glob

base_dir = './images/'

labels = [
    'ardido',
    'brocado',
    'marinheiro',
    'normal',
    'preto',
    'verde'
]

for label in labels:
    print(f'Loading data from: {label}')

    addrs = glob.glob(os.path.join(base_dir + label, '*.json'))

    for addr in addrs:
        with open(addr, 'r+') as json_file:
            data = json.load(json_file)

        for bean in data:
            bean['label'] = label

        with open(addr, 'w+') as f:
            json.dump(data, f, indent=2)
