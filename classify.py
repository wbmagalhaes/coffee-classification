import numpy as np

from CoffeeNet import create_model

from utils import data_reader, visualize

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


def decide(preds, threshold=0.5):
    conf = np.max(preds)
    if conf > threshold:
        return int(np.argmax(preds))
    else:
        return 6


y_pred = [decide(pred, threshold=0.48) for pred in y_pred]

defects_pred = visualize.count_defects(y_pred)
defects_true = visualize.count_defects(y_true)
print('pred', defects_pred)
print('true', defects_true)

total_pred = visualize.sum_defects(defects_pred)
total_true = visualize.sum_defects(defects_true)
print('pred', total_pred)
print('true', total_true)

visualize.plot_images(x_data[:100], y_true[:100], y_pred[:100])

# y_pred = np.argmax(y_pred, axis=1)
y_pred = np.append(y_pred, [3])
y_true = np.append(y_true, [3])

visualize.plot_confusion_matrix(y_true, y_pred, normalize=True)
