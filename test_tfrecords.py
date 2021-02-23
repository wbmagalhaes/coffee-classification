from utils import tfrecords, visualize, reload_model

model_name = 'CoffeeNet6'
epoch = 0

dataset = tfrecords.read(['./data/classification_test.tfrecord'])

x_data, y_true = zip(*[data for data in dataset])

model = reload_model.from_json(model_name, epoch)

_, y_pred = model.predict(dataset.batch(32))

visualize.plot_images(x_data[:64], y_true[:64], y_pred[:64], fontsize=8)
visualize.plot_confusion_matrix(y_true, y_pred)
