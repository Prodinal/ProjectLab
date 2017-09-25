import tensorflow as tf

LABELS = []

# used to examine whether classification is possible, e.g. are the values distinct enough for classification
def featureMapOfSlice(image_data):
    with tf.Session() as sess:
        featureTensor = sess.graph.get_tensor_by_name('input/BottleneckInputPlaceholder:0')

        featureMap = sess.run(featureTensor, {'DecodeJpeg:0': image_data})
        return featureMap


def evaluateSlice(image_data):
    with tf.Session() as sess:
        # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        # tensor_list = [n.name for n in tf.get_default_graph().as_graph_def().node]

        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg:0': image_data})

        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = LABELS[node_id]
            score = predictions[0][node_id]
            print('%s (score = % 5f)' % (human_string, score))
        return predictions


def init_tf(saved_graph_path, label_lines):
    global LABELS
    LABELS = label_lines
    with tf.gfile.FastGFile(saved_graph_path) as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

