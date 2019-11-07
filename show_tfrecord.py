from utils import tfrecords, augmentation, other, visualize


def show(filenames, batch=64, augment=True):
    dataset = tfrecords.read(filenames).shuffle(buffer_size=10000)
    dataset = dataset.map(other.normalize)
    visualize.plot_dataset(dataset.batch(batch))

    if augment:
        dataset = augmentation.apply(dataset)
        visualize.plot_dataset(dataset.batch(batch))


def main():
    filenames = ['./data/data_train.tfrecord']
    show(filenames)


if __name__ == "__main__":
    main()
