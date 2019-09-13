import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

from utils import config

model_id = 'CoffeeNet6_gap_dense_64'
checkpoint = 50000

export_dir = 'saved_models/' + model_id + '/'
training_dir = config.CHECKPOINT_DIR + model_id

print(f'Using model {model_id}')

clean_graph_def = None

with tf.Session(graph=tf.Graph()) as sess:
    ckpt = f'{training_dir}/model-{checkpoint}.meta'
    saver = tf.train.import_meta_graph(ckpt, clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(training_dir))
    print('Model loaded.')

    graph_def = tf.get_default_graph().as_graph_def()

    print(f"{len(graph_def.node)} ops in the graph.")

    clean_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=graph_def,
        output_node_names=['result/label', 'result/probs']
    )

    print(f"{len(clean_graph_def.node)} ops in the final graph.")

with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(clean_graph_def, name='')

    graph = tf.get_default_graph()
    graph_def = tf.get_default_graph().as_graph_def()

    for op in graph.get_operations():
        print(op.name)

    print('Saving model.')

    image = graph.get_tensor_by_name('inputs/img_input:0')

    label = graph.get_tensor_by_name('result/label:0')
    probs = graph.get_tensor_by_name('result/probs:0')

    inputs = {
        'img_input': image
    }

    outputs = {
        'label': label,
        'probs': probs
    }

    signature = predict_signature_def(inputs=inputs, outputs=outputs)

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={'predict': signature},
        clear_devices=True,
        strip_default_attrs=True
    )

    builder.save()

    print('Model saved.')
