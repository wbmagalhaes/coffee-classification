import numpy as np

from CoffeeNet6 import create_model

from utils import data_reader, visualize
from utils.labelmap import label_names

sample_paths = [
    'C:/Users/Usuario/Desktop/cafe_imgs/cut_samples/84A',
    'C:/Users/Usuario/Desktop/cafe_imgs/cut_samples/248A'
]

data = data_reader.load(sample_paths)

x_data, y_true = zip(*data)
x_data = np.array(x_data).astype(np.float32) / 255.
y_true = np.array(y_true).astype(np.float32)

model = create_model()
model.load_weights('./results/coffeenet6.h5')

_, y_pred = model.predict(x_data)

visualize.plot_images(x_data[:16], y_true[:16], y_pred[:16])

y_pred = np.argmax(y_pred, axis=1)
y_pred = np.append(y_pred, [3])
y_true = np.append(y_true, [3])

visualize.plot_confusion_matrix(y_true, y_pred, normalize=True)
