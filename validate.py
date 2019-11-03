from CoffeeNet import create_model

from utils import tfrecords, other, visualize

dataset = tfrecords.read(['./data/data_test.tfrecord'])
dataset = dataset.map(other.normalize)

x_data, y_true = zip(*[data for data in dataset])

model = create_model()
model.load_weights('./results/coffeenet6.h5')

_, y_pred = model.predict(dataset.batch(32))

visualize.plot_images(x_data[:64], y_true[:64], y_pred[:64], fontsize=8)
visualize.plot_confusion_matrix(y_true, y_pred)
