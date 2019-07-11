import os

import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt

from utils import config
from utils import labelmap


def read_xml(img_dir, addr):
    """Abre o arquivo xml, carrega a imagem e retorna as imagens cortadas e as labels
        Args:
            img_dir: diretório onde estão as imagens.
            addr: endereço do arquivo xml.

        Returns:
            string com o nome do arquivo da imagem
            lista de numpy arrays no formato (config.IMG_SIZE, config.IMG_SIZE, 3) com as imagens
            lista de inteiros com a label de cada imagem

    """
    tree = ET.parse(addr)
    root = tree.getroot()

    filename = root.find('filename').text
    print('Lendo imagem: ' + filename)

    image = cv.imread(os.path.join(img_dir, filename))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    
    height, width, _ = image.shape

    imgs = []
    labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text

        bndbox = obj.find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)

        size_x = abs(xmax - xmin)
        size_y = abs(ymax - ymin)
        size = int(max(size_x, size_y) / 2)

        center_x = int(size_x / 2) + xmin
        center_y = int(size_y / 2) + ymin

        xmin = max(center_x - size, 0)
        ymin = max(center_y - size, 0)

        xmax = min(center_x + size, width - 1)
        ymax = min(center_y + size, height -1 )

        croped = image[ymin:ymax, xmin:xmax]
        croped = cv.resize(croped, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv.INTER_AREA)
        # cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

        imgs.append(croped)

        label = labelmap.index_of_label(name)
        labels.append(label)

    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(image)
    # plt.show()

    return filename, imgs, labels
