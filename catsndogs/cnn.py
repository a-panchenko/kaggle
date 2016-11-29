import datetime

import tensorflow as tf
import time
import numpy as np
import os

import cnn_input


tf.app.flags.DEFINE_integer('batch_size', 256, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('train_dir', 'train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 60000, '')

FLAGS = tf.flags.FLAGS


def __conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def __maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def cnn(x):
    with tf.variable_scope('conv1'):
        w = tf.Variable(tf.random_normal([5, 5, 3, 64]))
        b = tf.Variable(tf.random_normal([64]))
        conv1 = __conv2d(x, w, b)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    with tf.variable_scope('conv2'):
        w = tf.Variable(tf.random_normal([5, 5, 64, 64]))
        b = tf.Variable(tf.random_normal([64]))
        conv2 = __conv2d(norm1, w, b)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        w = tf.Variable(tf.random_normal([dim, 1024], stddev=0.04))
        b = tf.Variable(tf.random_normal([1024]))
        local3 = tf.nn.relu(tf.matmul(reshape, w) + b, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        w = tf.Variable(tf.random_normal([1024, 192], stddev=0.04))
        b = tf.Variable(tf.random_normal([192]))
        local4 = tf.nn.relu(tf.matmul(local3, w) + b, name=scope.name)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        w = tf.Variable(tf.random_normal([192, 1], stddev=1 / 192.0))
        b = tf.Variable(tf.random_normal([1]))
        softmax_linear = tf.add(tf.matmul(local4, w), b, name=scope.name)
    return tf.nn.sigmoid(softmax_linear)


def loss(predicted, labels):
    labels = tf.cast(labels, tf.float32)
    c = tf.nn.l2_loss(predicted - labels)
    tf.add_to_collection('losses', c)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def optimize(total_loss, global_step):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
    optimizer.minimize(total_loss, global_step)
    return tf.no_op()


def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        images, labels = cnn_input.inputs(['data/exp/train_batch2'], tf.app.flags.FLAGS.batch_size, True)
        predicted = cnn(images)
        cost = loss(predicted, labels)

        train_op = optimize(cost, global_step)
        saver = tf.train.Saver(tf.all_variables())

        accuracy = tf.reduce_mean(tf.abs(predicted - tf.cast(labels, tf.float32)))

        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)

        tf.train.start_queue_runners(sess)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value, acc, p, l = sess.run([train_op, cost, accuracy, predicted, labels])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 100 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, accuracy = %.2f, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.datetime.now(), step, acc, loss_value,
                                    examples_per_sec, sec_per_batch))
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train()
