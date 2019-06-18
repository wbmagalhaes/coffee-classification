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
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.uint8)

    imgs = []
    labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text

        bndbox = obj.find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)

        sx = xmax - xmin
        sy = ymax - ymin

        s = max(sx, sy)

        h, w, _ = image.shape
        s = min(s, h, w)

        bx = int((s - sx) / 2)
        by = int((s - sy) / 2)

        xmin -= bx
        ymin -= by
        xmax += bx
        ymax += by

        croped = image[ymin:ymax, xmin:xmax]
        croped = cv.resize(croped, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv.INTER_AREA)

        # plt.imshow(croped)
        # plt.show()

        imgs.append(croped)

        label = labelmap.index_of_label(name)
        labels.append(label)

    return filename, imgs, labels
