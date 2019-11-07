from utils import tfrecords, other, visualize, reload_model


def start(dataset_filenames, model_name, epoch, batch=32, plot_num=64, fontsize=8, normalize_cm=False):
    dataset = tfrecords.read(dataset_filenames)
    dataset = dataset.map(other.normalize)

    x_data, y_true = zip(*[data for data in dataset])

    model = reload_model.from_json(model_name, epoch)

    _, y_pred = model.predict(dataset.batch(batch))

    visualize.plot_images(x_data[:plot_num], y_true[:plot_num], y_pred[:plot_num], fontsize=fontsize)
    visualize.plot_confusion_matrix(y_true, y_pred, normalize=normalize_cm)


def main():
    dataset_filenames = ['./data/data_test.tfrecord']
    model_name = 'CoffeeNet6'
    epoch = 0

    start(dataset_filenames, model_name, epoch)


if __name__ == "__main__":
    main()
