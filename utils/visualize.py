import tensorflow as tf
import numpy as np
import cv2
import math

from utils import labelmap

rect_h = 120
bordersize = 3
img_resized = 180


def show_images_compare(imgs, pred_labels, pred_confs, correct_labels, num, name):
    imgs = imgs[:min(num, len(imgs))]

    image_per_row = int(math.sqrt(num))
    n_row = math.ceil(num / image_per_row)

    imagelist = []
    for i in range(0, n_row):
        img_row = imgs[i *
                       image_per_row:min((i + 1) * image_per_row, len(imgs))]
        pred_row = pred_labels[i *
                               image_per_row:min((i + 1) * image_per_row, len(pred_labels))]
        conf_row = pred_confs[i *
                              image_per_row:min((i + 1) * image_per_row, len(pred_confs))]
        correct_row = correct_labels[i * image_per_row:min(
            (i + 1) * image_per_row, len(correct_labels))]

        row = image_row_compare(
            img_row, pred_row, conf_row, correct_row, image_per_row)
        imagelist.append(row)

    img = np.concatenate(imagelist, axis=0)
    img = cv2.copyMakeBorder(img, top=bordersize, bottom=0, left=bordersize,
                             right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    cv2.imwrite(name, img)


def show_images(imgs, pred_labels, pred_confs, num, name):
    imgs = imgs[:min(num, len(imgs))]

    image_per_row = int(math.sqrt(num))
    n_row = math.ceil(num / image_per_row)

    imagelist = []
    for i in range(0, n_row):
        img_row = imgs[i *
                       image_per_row:min((i + 1) * image_per_row, len(imgs))]
        pred_row = pred_labels[i *
                               image_per_row:min((i + 1) * image_per_row, len(pred_labels))]
        conf_row = pred_confs[i *
                              image_per_row:min((i + 1) * image_per_row, len(pred_confs))]

        row = image_row(img_row, pred_row, conf_row, image_per_row)
        imagelist.append(row)

    img = np.concatenate(imagelist, axis=0)
    img = cv2.copyMakeBorder(img, top=bordersize, bottom=0, left=bordersize,
                             right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    cv2.imwrite(name, img)


def image_row_compare(imgs, pred_labels, pred_confs, correct_labels, num):
    row = []
    for i in range(0, num):
        if i < len(imgs):
            img = imgs[i]
            pred_label = pred_labels[i]
            pred_conf = pred_confs[i]
            correct_label = np.argmax(correct_labels[i])
            row.append(image_with_label_compare(
                img, pred_label, pred_conf, correct_label))
        else:
            row.append(np.zeros((img_resized + rect_h + bordersize,
                                 img_resized + bordersize, 3), np.uint8))

    return np.concatenate(row, axis=1)


def image_row(imgs, pred_labels, pred_confs, num):
    row = []
    for i in range(0, num):
        if i < len(imgs):
            img = imgs[i]
            pred_label = pred_labels[i]
            pred_conf = pred_confs[i]
            row.append(image_with_label(img, pred_label, pred_conf))

        else:
            row.append(np.zeros((img_resized + rect_h + bordersize,
                                 img_resized + bordersize, 3), np.uint8))

    return np.concatenate(row, axis=1)


def image_with_label_compare(img, pred_label, pred_conf, correct_label):
    img = cv2.resize(img, (img_resized, img_resized), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = img.astype(np.uint8)

    result = np.ones((img_resized, img_resized, 3), np.uint8) * 255
    result[:img_resized, :img_resized, :3] = img

    result = cv2.copyMakeBorder(result, top=0, bottom=rect_h, left=0,
                                right=0, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .6
    lineType = 1
    fontColor = (0, 0, 0)

    bottomLeftCornerOfText = (10, img_resized + rect_h - 25)

    text = '{}'.format(labelmap.labels[correct_label]['name'])
    cv2.putText(result, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    if pred_label == correct_label:
        fontColor = (50, 255, 50)
    else:
        fontColor = (50, 50, 255)
        #print('= errado =')
        # for i in range(0, len(pred_conf)):
        #    print('{}: {:.1f}%'.format(labelmap.labels[i], pred_conf[i] * 100))

    bottomLeftCornerOfText = (10, img_resized + rect_h - 5)

    text = '{} {:.1f}%'.format(
        labelmap.labels[pred_label]['name'], pred_conf[pred_label] * 100)
    cv2.putText(result, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    result = cv2.copyMakeBorder(result, top=0, bottom=bordersize, left=0,
                                right=bordersize, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return result


def image_with_label(img, pred_label, pred_conf):
    img = cv2.resize(img, (img_resized, img_resized), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = img.astype(np.uint8)

    result = np.ones((img_resized, img_resized, 3), np.uint8) * 255
    result[:img_resized, :img_resized, :3] = img

    result = cv2.copyMakeBorder(result, top=0, bottom=rect_h, left=0,
                                right=0, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .6
    lineType = 1

    pred = np.argmax(pred_conf)
    idx = np.argsort(pred_conf)
    h = img_resized + rect_h - 5

    for i in idx:
        if i == pred:
            fontColor = (40, 200, 40)
        else:
            fontColor = (40, 40, 200)

        text = '{} {:.1f}%'.format(
            labelmap.labels[i]['name'], pred_conf[i] * 100)
        bottomLeftCornerOfText = (10, h)
        cv2.putText(result, text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        h -= 20

    result = cv2.copyMakeBorder(result, top=0, bottom=bordersize, left=0,
                                right=bordersize, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return result


def show_accuracy(pred_labels, correct_labels):
    total_preds = len(pred_labels)
    total_errors = 0

    label_counter = []

    correct_counter = []
    false_pos_counter = []

    for i in range(labelmap.count):
        label_counter.append(0)
        correct_counter.append(0)
        false_pos_counter.append(0)

    for i in range(total_preds):
        p_label = pred_labels[i]
        c_label = np.argmax(correct_labels[i])

        label_counter[c_label] += 1

        if c_label == p_label:
            correct_counter[c_label] += 1
        else:
            false_pos_counter[p_label] += 1
            total_errors += 1

    print(' ')
    print('==================')
    correct = total_preds - total_errors
    accuracy = correct * 100 / total_preds
    print('Acertou {:.2f}% ({} de {})'.format(accuracy, correct, total_preds))

    for i in range(labelmap.count):
        label = labelmap.labels[i]['name']
        print('===== {} ====='.format(label))

        total = label_counter[i]
        correct = correct_counter[i]
        accuracy = 0
        if total != 0:
            accuracy = correct * 100 / total

        print('Acertou {:.2f}% ({} de {})'.format(accuracy, correct, total))

        false_pos = 0
        if total_errors != 0:
            false_pos = false_pos_counter[i] * 100 / total_errors
        print('Falso positivo: {:.2f}% ({} de {})'.format(
            false_pos, false_pos_counter[i], total_errors))
