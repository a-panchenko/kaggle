import datetime

import tensorflow as tf
import time
import numpy as np
import os
import csv
import logging

import cnn_input


tf.app.flags.DEFINE_integer('batch_size', 16, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('train_dir', 'train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000, '')

FLAGS = tf.flags.FLAGS
LOG = logging.getLogger("cnn_logger")


def perform_weight_decay(var, wd):
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return var


def __conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def cnn(x):
    with tf.variable_scope('conv1'):
        w = tf.Variable(tf.random_normal([5, 5, 3, 64], stddev=0.05))
        b = tf.Variable(tf.random_normal([64]))
        conv1 = __conv2d(x, w, b)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    with tf.variable_scope('conv2'):
        w = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev=0.05))
        b = tf.Variable(tf.random_normal([128]))
        conv2 = __conv2d(norm1, w, b)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local1
    with tf.variable_scope('local1') as scope:
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        w = tf.Variable(tf.random_normal([dim, 1024], stddev=0.04))
        w = perform_weight_decay(w, 0.004)
        b = tf.Variable(tf.random_normal([1024]))
        local1 = tf.nn.relu(tf.matmul(reshape, w) + b, name=scope.name)

    with tf.variable_scope('local2') as scope:
        w = tf.Variable(tf.random_normal([1024, 384]))
        b = tf.Variable(tf.random_normal([384]))
        w = perform_weight_decay(w, 0.004)
        local2 = tf.nn.relu(tf.matmul(local1, w) + b, name=scope.name)

    with tf.variable_scope('local3') as scope:
        w = tf.Variable(tf.random_normal([384, 192]))
        b = tf.Variable(tf.random_normal([192]))
        w = perform_weight_decay(w, 0.004)
        local3 = tf.nn.relu(tf.matmul(local2, w) + b, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        w = tf.Variable(tf.random_normal([192, 1], stddev=1.0/192.0))
        b = tf.Variable(tf.random_normal([1]))
        fc = tf.add(tf.matmul(local3, w), b, name=scope.name)
        # fc = tf.nn.dropout(fc, 0.5)
    return tf.nn.sigmoid(fc)


def loss(logits, labels):
    labels = tf.cast(labels, tf.float32)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # cross_entropy_mean = tf.reduce_mean(tf.abs(labels - logits))
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def optimize(total_loss, global_step):
    tf.train.AdamOptimizer(learning_rate=0.01).minimize(total_loss, global_step)
    return tf.no_op()


def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images, labels = cnn_input.inputs(['data/train_batch_64_not_random'], tf.app.flags.FLAGS.batch_size, train=True)

        logits = cnn(images)

        l = loss(logits, labels)
        train_op = optimize(l, global_step)

        diff = tf.ones_like(logits) - tf.abs((tf.cast(labels, tf.float32) - logits))
        accuracy = tf.reduce_mean(diff)

        saver = tf.train.Saver(tf.global_variables())

        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        tf.train.start_queue_runners(sess)

        sum_cost = 0
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value, acc, lb, lg, im = sess.run([train_op, l, accuracy, labels, logits, images])
            duration = time.time() - start_time
            sum_cost += loss_value
            avg_cost = sum_cost / (step+1)

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 100 == 0:
                # print(im)
                # print(lb)
                # print(lg)
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, accuracy = %.7f, loss = %.7f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                log_msg = format_str % (datetime.datetime.now(), step, acc, avg_cost,
                                    examples_per_sec, sec_per_batch)
                print(log_msg)
                LOG.debug(log_msg)
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def predict():
    with tf.Graph().as_default(), open('../output/predicted.csv', 'w+') as csvfile:
        fieldnames = ['id', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'id': 'id', 'label': 'label'})
        images = cnn_input.inputs(['data/predict_batch_64_not_random'], tf.app.flags.FLAGS.batch_size, train=False)
        predicted = cnn(images)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session()
        tf.train.start_queue_runners(sess)
        sess.run(init)
        for i in range(1, 12501):
            if i % 1000 == 0:
                print(i)
            ckpt = tf.train.get_checkpoint_state('train')
            model_checkpoint_path = 'train/model.ckpt-30000'
            saver.restore(sess,  model_checkpoint_path)
            # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

            probability = sess.run([predicted])
            label = probability[0][0][0]
            writer.writerow({'id': str(i), 'label': str(label)})


if __name__ == '__main__':
    logging.basicConfig(filename="logs.log",
                        format='%(asctime)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s',
                        level=logging.DEBUG)
    train()
    # predict()
