import tensorflow as tf
from functools import reduce


def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_decoded = tf.cast(image_decoded, tf.float32)/255

    label = tf.read_file(label)
    label = tf.image.decode_jpeg(label)
    label = tf.cast(label, tf.float32)/255

    return image_decoded, label


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d1(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv2d2(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def leaky_relu(inputs, slope=0.2):
    return tf.maximum(slope * inputs, inputs)


def instance_norm(net):
    mean, var = tf.nn.moments(net, axes=[1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(mean.shape[-1]))
    scale = tf.Variable(tf.ones(mean.shape[-1]))
    return (net - mean) * scale / (tf.sqrt(var + 1e-3)) + shift


def get_param(program, prefix):
    all_params = program.global_block().all_parameters()
    return [t.name for t in all_params if t.name.startswith(prefix)]


def tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)