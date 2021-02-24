import os
import glob
from model import *
from vgg19 import *

img_H = 256
img_W = 256
img_C = 1
batchsize = 64
epoch_time = 100
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

distortion = tf.placeholder(tf.float32, [batchsize, img_H, img_W, img_C], name="distortion")
label = tf.placeholder(tf.float32, [batchsize, img_H, img_W, img_C], name="label")

fake = Generator(distortion)
fake_logit = Discriminator(fake, False)
real_logit = Discriminator(label, True)

e = tf.random_uniform([batchsize, 1, 1, 1], 0, 1)
x_hat = e * label + (1 - e) * fake
zzz = Discriminator(x_hat, True)
grad = tf.gradients(zzz, x_hat)[0]

with tf.name_scope("d_loss"):
    d_loss = abs(tf.reduce_mean(fake_logit - real_logit) + 10 * tf.reduce_mean(
        tf.square(tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])) - 1)))

with tf.name_scope("tvloss"):
    batch_shape = (batchsize, img_W, img_H, 1)
    tv_y_size = tensor_size(fake[:, 1:, :, :])
    tv_x_size = tensor_size(fake[:, :, 1:, :])
    y_tv = tf.nn.l2_loss(fake[:, 1:, :, :] - fake[:, :batch_shape[1] - 1, :, :])
    x_tv = tf.nn.l2_loss(fake[:, :, 1:, :] - fake[:, :, :batch_shape[2] - 1, :])
    loss_tv = 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batchsize

with tf.name_scope("gloss"):
    g_loss = tf.reduce_mean(-fake_logit)

with tf.name_scope("contentloss"):
    CONTENT_LAYER = 'conv5_4'
    fake_3 = tf.concat([fake, fake, fake], 3)
    label_3 = tf.concat([label, label, label], 3)
    enhanced_vgg = net(preprocess(fake_3 * 255))
    dslr_vgg = net(preprocess(label_3 * 255))
    content_size = tensor_size(dslr_vgg[CONTENT_LAYER]) * batchsize
    loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size


with tf.name_scope("l1loss"):
    loss_l1 = tf.reduce_mean(tf.abs(fake * 255 - label * 255))

with tf.name_scope("sum_loss"):
    loss_generator = 1 * g_loss + loss_content + 500 * loss_tv + 3 * loss_l1

generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
discriminator_vars = [v for v in tf.global_variables() if v.name.startswith("discriminator")]

optim = tf.train.AdamOptimizer(5e-5)
opt_D = optim.minimize(d_loss, var_list=discriminator_vars)
opt_G = optim.minimize(loss_generator, var_list=generator_vars)

img_path = tf.constant(glob.glob('./distortion/*.jpg'))
gt_path = tf.constant(glob.glob('./label/*.jpg'))
length = len(glob.glob('./label/*.jpg'))
filename_queue = tf.data.Dataset.from_tensor_slices((img_path, gt_path))
dataset = filename_queue.map(parse_function)
dataset = dataset.batch(batchsize, drop_remainder=True).repeat(epoch_time)
iterator = dataset.make_initializable_iterator()
inte = iterator.get_next()
init_op = iterator.make_initializer(dataset)

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('./para/')
    if ckpt is not None:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("No model!")

    epoch, batch = 0, 0
    while True:
        try:
            images, labels = sess.run(inte)
            images = np.reshape(images, [batchsize, 256, 256, img_C])
            labels = np.reshape(labels, [batchsize, 256, 256, img_C])
            [dloss, _] = sess.run([d_loss, opt_D], feed_dict={distortion: images, label: labels})
            [loss_temp, _, content, l1, tv] = sess.run([loss_generator, opt_G, loss_content, loss_l1, loss_tv],
                feed_dict={distortion: images, label: labels})

            batch += 1
            if batch % 100 == 0:
                print("epoch:%g, batch:%g, sumloss:%g, contentloss:%g, l1loss:%g, tvloss:%g, dloss:%g" %
                      (epoch, batch, loss_temp, content, l1, tv, dloss))

            if batch == length // batchsize:
                epoch += 1
                batch = 0
                if epoch == 100:
                    saver.save(sess, "./para/model.ckpt", write_meta_graph=False)

        except tf.errors.OutOfRangeError:
            break
