import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_data(arg_feature_count, arg_output_file=None):
    result_a = np.random.rand(10, arg_feature_count) + 1
    result_b = np.random.rand(10, arg_feature_count)
    plt.scatter(result_a[:, 0], result_a[:, 1], c='r', marker='x')
    plt.scatter(result_b[:, 0], result_b[:, 1], c='g', marker='o')
    if arg_output_file is None:
        plt.show()
    else:
        plt.savefig(arg_output_file)
    return result_a, result_b


if __name__ == '__main__':
    start_time = time.time()

    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    feature_count = 2
    hidden_count = 10
    random_seed = 1
    np.random.seed(random_seed)
    scatter_plot_file = './output/page227-scatter.png'
    logger.debug('writing scatter plot to %s' % scatter_plot_file)
    data_a, data_b = get_data(arg_feature_count=feature_count, arg_output_file=scatter_plot_file)

    with tf.name_scope('input'):
        x1 = tf.placeholder(tf.float32, [None, feature_count], name='x1')
        x2 = tf.placeholder(tf.float32, [None, feature_count], name='x2')
        dropout_keep_probability = tf.placeholder(tf.float32, name='dropout_probability')

    with tf.name_scope('hidden_layer'):
        with tf.name_scope('weights'):
            w1 = tf.Variable(tf.random_normal([feature_count, hidden_count]), name='w1')
            tf.summary.histogram('w1', w1)
            b1 = tf.Variable(tf.random_normal([hidden_count]), name='b1')
            tf.summary.histogram('b1', b1)

        with tf.name_scope('output'):
            h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x1, w1) + b1), keep_prob=dropout_keep_probability)
            tf.summary.histogram('h1', h1)
            h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(x2, w1) + b1), keep_prob=dropout_keep_probability)
            tf.summary.histogram('h2', h2)

    with tf.name_scope('output_layer'):
        with tf.name_scope('output_layer'):
            w2 = tf.Variable(tf.random_normal([hidden_count, 1]), name='w2')
            tf.summary.histogram('w2', w2)
            b2 = tf.Variable(tf.random_normal([1]), name='b2')
            tf.summary.histogram('b2', b2)
        with tf.name_scope('output'):
            s1 = tf.matmul(h1, w2) + b2
            s2 = tf.matmul(h2, w2) + b2

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
