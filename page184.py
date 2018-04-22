import logging
import time

import tensorflow as tf

from page174a import read_data

start_time = time.time()


# todo try rolling this up into one line
def conv_layer(arg_x, arg_W, arg_b):
    conv = tf.nn.conv2d(arg_x, arg_W, strides=[1, 1, 1, 1], padding='SAME')
    conv_with_b = tf.nn.bias_add(conv, arg_b)
    result = tf.nn.relu(conv_with_b)
    return result


if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    names, data, labels = read_data('./cifar-10-batches-py', arg_logger=logger)
    x = tf.placeholder(tf.float32, (None, 2 * 24))
    y = tf.placeholder(tf.float32, (None, len(names)))
    W1 = tf.Variable(tf.random_normal([5, 5, 1, 64]))
    b1 = tf.Variable(tf.random_normal([64]))
    W2 = tf.Variable(tf.random_normal([5, 5, 64, 64]))
    b2 = tf.Variable(tf.random_normal([64]))
    W3 = tf.Variable(tf.random_normal([6 * 6 * 64, 1024]))
    b3 = tf.Variable(tf.random_normal([1024]))
    W_out = tf.Variable(tf.random_normal([1024, len(names)]))
    b_out = tf.Variable(tf.random_normal([len(names)]))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
