import common
import tensorflow as tf

__all__ = ['WINDOW_SHAPE',
           'get_training_model',
           'get_detect_model']


WINDOW_SHAPE = (128, 128)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)


def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1],
                          padding='SAME')


def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x,
                          ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1],
                          padding='SAME')


def convolutional_layers():
    # Second and third dimensions are width, height, first is number of image
    x = tf.placeholder(tf.float32, [None, None, None])

    # First layer

    # 5by5 convolutional kernel, 1 input image, 48 outputs
    W_conv1 = weight_variable([5, 5, 1, 48])
    b_conv1 = bias_variable([48])

    # So it can be multiplied by the weight matrix
    # x_expanded = tf.expand_dims(tf.expand_dims(x, 2), 3)
    x_expanded = tf.expand_dims(x, 3)
    # print(tf.shape(x))
    # x_expanded = tf.reshape(x, [-1, 128, 128, 1])
    
    h_conv1 = tf.nn.relu(conv2d(x_expanded, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))

    # Second layer
    W_conv2 = weight_variable([5, 5, 48, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(2, 2), stride=(2, 2))

    # Third layer
    W_conv3 = weight_variable([5,5,64,128])
    b_conv3 = bias_variable([128])


    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))

    return x, h_pool3, [W_conv1, b_conv1,
                        W_conv2, b_conv2, 
                        W_conv3, b_conv3]


def get_training_model():

    x, conv_layer, conv_vars = convolutional_layers()

    # Densely connected convolutional_layers
    # 128 image times 16by16 "pixel" input, 2048 outputs (=neurons)
    W_fc1 = weight_variable([16*16*128, 2048])
    b_fc1 = bias_variable([2048])

    conv_layer_flat = tf.reshape(conv_layer, [-1, 16*16*128])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)

    # Output Layer
    W_fc2 = weight_variable([2048, len(common.OUTCOMES)])
    b_fc2 = bias_variable([len(common.OUTCOMES)])
    
    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    return (x, y, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])

def get_detect_model():

    x, conv_layer, conv_vars = convolutional_layers()

    # Fourth layer
    W_fc1 = weight_variable([16*16*128, 2048])
    W_conv1 = tf.reshape(W_fc1, [16, 16, 128, 2048])

    b_fc1 = bias_variable([2048])

    h_conv1 = tf.nn.relu(conv2d(conv_layer, W_conv1,
                        stride=(1, 1), padding='VALID') + b_fc1)

    # Fifth layer
    W_fc2 = weight_variable([2048, len(common.OUTCOMES)])
    # Not sure about the  "1, 1,"
    W_conv2 = tf.reshape(W_fc2, [1, 1, 2048, len(common.OUTCOMES)])
    b_fc2 = bias_variable([len(common.OUTCOMES)])
    
    h_conv2 = conv2d(h_conv1, W_conv2) + b_fc2

    return (x, h_conv2, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])
    
