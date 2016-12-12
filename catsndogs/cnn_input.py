import tensorflow as tf
import numpy as np

IMAGE_SIZE = 64


NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 12500


def read_image(filename_queue):
    class ImageRecord(object):
        pass

    result = ImageRecord()

    label_bytes = 1
    result.height = IMAGE_SIZE
    result.width = IMAGE_SIZE
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)
    # label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def read_image_for_predicting(filename_queue):
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    depth = 3
    record_bytes = height * width * depth
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    depth_major = tf.reshape(record_bytes, [depth, height, width])

    return tf.transpose(depth_major, [1, 2, 0])


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 4
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
        return images, label_batch
    else:
        images = tf.train.batch(
            [image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
        return images


def inputs(data_files, batch_size, train):
    for f in data_files:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    filename_queue = tf.train.string_input_producer(data_files)
    if train:
        image = read_image(filename_queue)
        reshaped_image = tf.cast(image.uint8image, tf.float32)
        reshaped_image = tf.image.random_brightness(reshaped_image, max_delta=23)
        reshaped_image = tf.image.random_contrast(reshaped_image, lower=0.5, upper=1.4)
        reshaped_image = tf.image.per_image_standardization(reshaped_image)
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

        # Generate a batch of images and labels by building up a queue of examples.
        print('Filling queue with %d images before starting.' % min_queue_examples)
        return _generate_image_and_label_batch(reshaped_image, image.label,
                                               min_queue_examples, batch_size,
                                               shuffle=True)
    else:
        image = read_image_for_predicting(filename_queue)
        reshaped_image = tf.cast(image, tf.float32)
        reshaped_image = tf.image.per_image_standardization(reshaped_image)
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)
        print('Filling queue with %d images before predicting.' % min_queue_examples)
        return _generate_image_and_label_batch(reshaped_image, None,
                                               min_queue_examples, batch_size,
                                               shuffle=False)
