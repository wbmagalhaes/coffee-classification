import numpy as np

from utils import data_reader, visualize, reload_model


def start(sample_paths, model_name, epoch, plot_num=64, fontsize=8, normalize_cm=False):
    data = data_reader.load(sample_paths)

    x_data, y_true = zip(*data)
    x_data = np.array(x_data).astype(np.float32) / 255.
    y_true = np.array(y_true).astype(np.float32)

    model = reload_model.from_json(model_name, epoch)

    _, y_pred = model.predict(x_data)

    defects_pred = visualize.count_defects(y_pred)
    defects_true = visualize.count_defects(y_true)
    print('pred', defects_pred)
    print('true', defects_true)

    total_pred = visualize.sum_defects(defects_pred)
    total_true = visualize.sum_defects(defects_true)
    print('pred', total_pred)
    print('true', total_true)

    visualize.plot_images(x_data[:plot_num], y_true[:plot_num], y_pred[:plot_num], fontsize=fontsize)

    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.append(y_pred, [3])
    y_true = np.append(y_true, [3])

    visualize.plot_confusion_matrix(y_true, y_pred, normalize=normalize_cm)


def main():
    sample_paths = [
        'C:/Users/Usuario/Desktop/cafe_imgs/cut_samples/84A',
        'C:/Users/Usuario/Desktop/cafe_imgs/cut_samples/248A'
    ]
    model_name = 'CoffeeNet6'
    epoch = 0

    start(sample_paths, model_name, epoch)


if __name__ == "__main__":
    main()
