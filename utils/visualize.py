import tensorflow as tf
import numpy as np
import cv2
import math

from utils import labelmap
import matplotlib.pyplot as plt
import itertools

rect_h = 250
bordersize = 2
img_resized = 200


def show_images_compare(imgs, pred_labels, pred_confs, correct_labels, num, name):
    imgs = imgs[:min(num, len(imgs))]

    image_per_row = int(math.sqrt(num))
    n_row = math.ceil(num / image_per_row)

    imagelist = []
    for i in range(n_row):
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
    for i in range(n_row):
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
    for i in range(num):
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
    for i in range(num):
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


def show_accuracy(pred_labels, correct_labels, ignore_label=-1):
    total_preds = 0
    total_errors = 0
    label_counter = [0] * labelmap.count
    correct_counter = [0] * labelmap.count
    false_pos_counter = [0] * labelmap.count

    for p_label, c_label in zip(pred_labels, correct_labels):
        if c_label == ignore_label:
            continue

        total_preds += 1
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


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='Blues'):
    """
    This function modified to plots the ConfusionMatrix object.
    Normalization can be applied by setting `normalize=True`.

    Code Reference : 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    plt_cm = []
    for i in cm.classes:
        row = []
        for j in cm.classes:
            row.append(cm.table[i][j])
        plt_cm.append(row)
    plt_cm = np.array(plt_cm)
    if normalize:
        plt_cm = plt_cm.astype('float') / plt_cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(plt_cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm.classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = plt_cm.max() / 2.
    for i, j in itertools.product(range(plt_cm.shape[0]), range(plt_cm.shape[1])):
        plt.text(j, i, format(plt_cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if plt_cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predict')
    plt.show()
