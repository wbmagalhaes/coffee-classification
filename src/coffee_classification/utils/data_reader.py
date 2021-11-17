import os
import glob
import json
import cv2


def open_images(images_dir):
    path = os.path.join(images_dir, '**/*.jpg')
    addrs = glob.glob(path, recursive=True)
    images = [open_image(addr) for addr in addrs]
    return images, addrs


def open_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def open_jsons(json_dir):
    path = os.path.join(json_dir, '**/*.json')
    addrs = glob.glob(path, recursive=True)
    data = [open_json(addr) for addr in addrs]
    return data, addrs


def open_json(addr):
    with open(addr, 'r') as f:
        data = json.load(f)
    return data


def save_jsons(datas, paths):
    for data, path in zip(datas, paths):
        save_json(data, path)


def save_json(data, path):
    with open(path, 'w+') as f:
        json.dump(data, f, indent=2)
