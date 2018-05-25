import logging
import time

import tensorflow as tf
from tensorflow.contrib import rnn


class SeriesPredictor:
    def __init__(self, input_dim, seq_size, hidden_dim=10):
        self.checkpoint_file = './lstm.ckpt'
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim

        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        self.saver = tf.train.Saver()

        self.formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
        self.logger = logging.getLogger('SeriesPredictor')
        self.logger.setLevel(logging.DEBUG)
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)
        self.console_handler.setLevel(logging.DEBUG)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.console_handler.close()
        self.logger.removeHandler(console_handler)

    def model(self):
        cell = rnn.BasicLSTMCell(self.hidden_dim)
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        result = tf.matmul(outputs, W_repeated) + self.b_out
        result = tf.squeeze(result)
        return result

    def train(self, train_x, train_y):
        with tf.Session() as session:
            tf.get_variable_scope().reuse_variables()
            session.run(tf.global_variables_initializer())
            for index in range(1000):
                feed_dict = {self.x: train_x, self.y: train_y}
                _, mse = session.run([self.train_op, self.cost], feed_dict=feed_dict)
                if index % 100 == 0:
                    self.logger.debug('train operation iteration %d has mse %.4f' % (index, mse))
            save_path = self.saver.save(session, self.checkpoint_file)
            self.logger.debug('Model saved to {}'.format(save_path))

    def test(self, arg_session, test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(arg_session, self.checkpoint_file)
        feed_dict = {self.x: test_x}
        result = arg_session.run(self.model(), feed_dict=feed_dict)
        self.logger.debug(result)
        return result


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

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
