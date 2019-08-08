import tensorflow as tf

model_id = 'CoffeeNet6_all_images'
model_dir = 'saved_models/' + model_id

input_arrays = ["inputs/img_input"]
output_arrays = ["result/label", 'result/probs']
input_shapes = {
    "inputs/img_input": [None, 64, 64, 3]
}

with tf.Session() as sess:
    converter = tf.lite.TFLiteConverter.from_saved_model(
        model_dir,
        input_arrays=input_arrays,
        input_shapes=input_shapes,
        output_arrays=output_arrays,
        signature_key="predict"
    )

    tflite_model = converter.convert()

    print(converter._input_tensors)
    print(converter._output_tensors)

    open("coffeenet6_v1_1.3.tflite", "wb").write(tflite_model)
