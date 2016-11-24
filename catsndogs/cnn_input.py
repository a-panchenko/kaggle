import tensorflow as tf


IMAGE_SIZE = 227

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_image(filename_queue):
    """Reads and parses examples from data files."""

    class ImageRecord(object):
        pass

    result = ImageRecord()

    label_bytes = 1
    result.height = 227
    result.width = 227
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels."""
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 4
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return images, tf.reshape(label_batch, [batch_size])


def inputs(data_files, batch_size, train):
    for f in data_files:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    filename_queue = tf.train.string_input_producer(data_files)

    image = read_image(filename_queue)
    reshaped_image = tf.cast(image.uint8image, tf.float32)

    if train:
        reshaped_image = tf.image.random_brightness(reshaped_image, max_delta=63)
        reshaped_image = tf.image.random_contrast(reshaped_image, lower=0.2, upper=1.8)
        reshaped_image = tf.image.per_image_whitening(reshaped_image)

    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN if train else NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    min_fraction_of_examples_in_queue = 0.4

    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d images. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(reshaped_image, image.label,
                                           min_queue_examples, batch_size,
                                           shuffle=train)

