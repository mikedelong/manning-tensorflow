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


def visualize_results(arg_data_test, arg_session, arg_s1, arg_size):
    plt.figure()
    scores_test = arg_session.run(arg_s1, feed_dict={x1: arg_data_test, dropout_keep_probability: 1.0})
    scores_image = np.reshape(scores_test, [arg_size, arg_size])
    plt.imshow(scores_image, origin='lower')
    plt.colorbar()
    plt.savefig('./output/page227-scores.png')


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

    with tf.name_scope('loss'):
        s12 = s1 - s2
        s12_flat = tf.reshape(s12, [-1])
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.zeros_like(s12_flat), logits=s12_flat + 1)
        loss = tf.reduce_mean(cross_entropy)

    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    session = tf.InteractiveSession()
    summary_operation = tf.summary.merge_all()
    writer = tf.summary.FileWriter('tensorboard_files', session.graph)
    initializer = tf.global_variables_initializer()
    session.run(initializer)

    epoch_count = 10000
    for epoch in range(0, epoch_count):
        loss_value, _ = session.run([loss, train_op],
                                    feed_dict={x1: data_a, x2: data_b, dropout_keep_probability: 0.5})
        if epoch % 100 == 0:
            summary_result = session.run(summary_operation,
                                         feed_dict={x1: data_a, x2: data_b, dropout_keep_probability: 1.0})
            writer.add_summary(summary_result, epoch)

    grid_size = 10
    data_test = [[x, y] for y in np.linspace(0.0, 1.0, num=grid_size) for x in np.linspace(0.0, 1.0, num=grid_size)]
    visualize_results(data_test, arg_session=session, arg_s1=s1, arg_size=grid_size)

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
