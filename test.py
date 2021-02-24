from model import *
import cv2
import numpy as np

h = 256
w = 256

x_ = tf.placeholder(tf.float32, [None, w*h*1])
x_image = tf.reshape(x_, [-1, h, w, 1])
enhanced = Generator(x_image)


with tf.Session() as sess:

    saver = tf.train.Saver()
    saver.restore(sess, "./para/" + "model.ckpt")

    for i in range(1000):
        I = cv2.imread("./test/distortion/"+str(i)+".jpg", cv2.IMREAD_GRAYSCALE)
        II = np.float32(np.reshape(I, [1, h, w, 1])) / 255
        enhanced_2d = sess.run(enhanced, feed_dict={x_image: II})
        crop1 = np.reshape(enhanced_2d, [h, w, 1])
        cv2.imwrite("res/" + str(i) + ".jpg", crop1 * 255)