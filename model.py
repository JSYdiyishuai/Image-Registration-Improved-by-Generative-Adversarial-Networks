from function import *


def Generator(input_image):
    with tf.variable_scope("generator"):

        w1 = weight_variable([9, 9, 1, 64], name="w1")
        b1 = bias_variable([64], name="b1")
        c1 = leaky_relu(conv2d1(input_image, w1) + b1)

        w2 = weight_variable([3, 3, 64, 64], name="w2")
        b2 = bias_variable([64], name="b2")
        c2 = leaky_relu(instance_norm(conv2d1(c1, w2) + b2))

        w3 = weight_variable([3, 3, 64, 64], name="w3")
        b3 = bias_variable([64], name="b3")
        c3 = leaky_relu(instance_norm(conv2d1(c2, w3) + b3)) + c1

        w4 = weight_variable([3, 3, 64, 64], name="w4")
        b4 = bias_variable([64], name="b4")
        c4 = leaky_relu(instance_norm(conv2d1(c3, w4) + b4))

        w5 = weight_variable([3, 3, 64, 64], name="w5")
        b5 = bias_variable([64], name="b5")
        c5 = leaky_relu(instance_norm(conv2d1(c4, w5) + b5)) + c3

        w6 = weight_variable([3, 3, 64, 64], name="w6")
        b6 = bias_variable([64], name="b6")
        c6 = leaky_relu(instance_norm(conv2d1(c5, w6) + b6))

        w7 = weight_variable([3, 3, 64, 64], name="w7")
        b7 = bias_variable([64], name="b7")
        c7 = leaky_relu(instance_norm(conv2d1(c6, w7) + b7)) + c5

        w8 = weight_variable([3, 3, 64, 64], name="w8")
        b8 = bias_variable([64], name="b8")
        c8 = leaky_relu(instance_norm(conv2d1(c7, w8) + b8))

        w9 = weight_variable([3, 3, 64, 64], name="w9")
        b9 = bias_variable([64], name="b9")
        c9 = leaky_relu(instance_norm(conv2d1(c8, w9) + b9)) + c7

        w10 = weight_variable([3, 3, 64, 64], name="w10")
        b10 = bias_variable([64], name="b10")
        c10 = leaky_relu(instance_norm(conv2d1(c9, w10) + b10))

        w11 = weight_variable([3, 3, 64, 64], name="w11")
        b11 = bias_variable([64], name="b11")
        c11 = leaky_relu(instance_norm(conv2d1(c10, w11) + b11))

        w13 = weight_variable([9, 9, 64, 1], name="w13")
        b13 = bias_variable([1], name="b13")
        c13 = tf.tanh(conv2d1(c11, w13) + b13)

    return c13


def Discriminator(input_image, reuse):
    with tf.variable_scope("discriminator", reuse=reuse):

        dw1 = weight_variable([9, 9, 1, 32], name="dw1")
        db1 = bias_variable([32], name="db1")
        conv1 = leaky_relu(instance_norm(tf.nn.conv2d(input_image, dw1, strides=[1, 4, 4, 1], padding='SAME') + db1))

        dw3 = weight_variable([7, 7, 32, 64], name="dw3")
        db3 = bias_variable([64], name="db3")
        conv3 = leaky_relu(instance_norm(conv2d2(conv1, dw3) + db3))

        dw5 = weight_variable([3, 3, 64, 128], name="dw5")
        db5 = bias_variable([128], name="db5")
        conv5 = leaky_relu(instance_norm(conv2d2(conv3, dw5) + db5))

        dw7 = weight_variable([3, 3, 128, 256], name="dw7")
        db7 = bias_variable([256], name="db7")
        conv7 = leaky_relu(instance_norm(conv2d2(conv5, dw7) + db7))

        flat_size = 256 * 8 * 8
        conv8_flat = tf.reshape(conv7, [-1, flat_size])

        with tf.name_scope("d_layer5"):
            W_fc = tf.Variable(tf.truncated_normal([flat_size, 512], stddev=0.01), name="fcw1")
            bias_fc = tf.Variable(tf.constant(0.01, shape=[512]), name="fcb1")
            fc = leaky_relu(tf.matmul(conv8_flat, W_fc) + bias_fc)
        with tf.name_scope("d_layer6"):
            W_out = tf.Variable(tf.truncated_normal([512, 1], stddev=0.01), name="fcw2")
            bias_out = tf.Variable(tf.constant(0.01, shape=[1]), name="fcb2")
            adv_out = tf.matmul(fc, W_out) + bias_out

    return adv_out