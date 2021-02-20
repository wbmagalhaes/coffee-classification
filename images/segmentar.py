import os
import cv2

import json
import glob
import numpy as np


class ColorSpace():
    RGB = 0
    GRAY = 1
    HSV = 2
    LAB = 3
    YUV = 4


def otsu(raw_img, colorSpace, channel, invert, opening_it, dilate_it, fg_threshold, erode_it):
    if colorSpace == ColorSpace.GRAY:
        input_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
    else:
        if colorSpace == ColorSpace.RGB:
            input_img = raw_img
        if colorSpace == ColorSpace.HSV:
            input_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2HSV)
        elif colorSpace == ColorSpace.LAB:
            input_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2LAB)
        elif colorSpace == ColorSpace.YUV:
            input_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2YUV)

        input_img = input_img[:, :, channel]

    blur = cv2.GaussianBlur(input_img, (5, 5), 0)

    if invert:
        thresh = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        thresh = cv2.THRESH_BINARY + cv2.THRESH_OTSU

    _, thresh = cv2.threshold(blur, 0, 255, thresh)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=opening_it)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=dilate_it)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    _, sure_fg = cv2.threshold(dist_transform, fg_threshold * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(raw_img, markers)
    thresh[markers == -1] = 0

    return input_img, cv2.erode(thresh, kernel, iterations=erode_it)


def findBeans(mask, expand, min_area, max_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    beans = []
    for contour in contours:
        # torna convexo
        contour = cv2.convexHull(contour)

        # separa xs e ys
        xs, ys = [], []
        for point in contour:
            x, y = point[0]
            xs.append(x)
            ys.append(y)

        xs = np.array(xs)
        ys = np.array(ys)

        # aumenta um pouco o corte pra compensar o espaço minimo entre os grãos
        x = int(sum(xs) / len(xs))
        y = int(sum(ys) / len(ys))
        xs = (xs - x) * expand + x
        ys = (ys - y) * expand + y

        xmin, xmax = int(min(xs)), int(max(xs))
        ymin, ymax = int(min(ys)), int(max(ys))

        # calcula o tamanho
        size_x = xmax - xmin
        size_y = ymax - ymin

        area = size_x * size_y
        if area <= min_area or area >= max_area:
            continue

        contour = [[x, y] for x, y in zip(xs, ys)]
        contour = np.array(contour, dtype=np.int32)
        beans.append(contour)

    return beans


addrs = glob.glob('**/*.jpg', recursive=True)
for addr in addrs:

    print(os.path.basename(addr))
    json_path = addr[:-3] + 'json'

    if os.path.isfile(json_path):
        print('Pulando imagem já segmentada.')
        continue

    raw_img = cv2.imread(path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    input_img, mask = otsu(raw_img, ColorSpace.LAB, 0, True, 5, 5, 0.7, 1)
    beans = findBeans(mask, 1.1, 200, 4000)

    data = []
    for bean in beans:

        points = []
        for point in bean:
            point = [float(point[0]), float(point[1])]
            points.append(point)

        data.append({
            "label": "não classificado",
            "points": points
        })

    with open(json_path, 'w+') as f:
        json.dump(data, f, indent=2)
