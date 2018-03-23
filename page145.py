import logging
import time

import numpy as np
import tensorflow as tf
from sklearn import datasets

start_time = time.time()

random_seed = 143

tf.set_random_seed(random_seed)


class Autoencoder:
    def __init__(self, input_dim, hidden_dim, epoch=250, learning_rate=0.001):
        self.epoch = epoch
        self.learning_rate = learning_rate
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
        with tf.name_scope('encode'):
            weights = tf.Variable(tf.random_normal([input_dim, hidden_dim], dtype=tf.float32, name='weights'))
            biases = tf.Variable(tf.zeros([hidden_dim]), name='biases')
            encoded = tf.nn.tanh(tf.matmul(x, weights) + biases)

        with tf.name_scope('decode'):
            weights = tf.Variable(tf.random_normal([hidden_dim, input_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([input_dim]), name='biases')
            decoded = tf.matmul(encoded, weights) + biases

        self.x = x
        self.encoded = encoded
        self.decoded = decoded

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded))))
        self.training_operation = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()
        self.model_file = './model.checkpoint'

    def train(self, arg_data, interval=10):
        num_samples = len(arg_data)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for index in range(self.epoch):
                current_loss = None
                for sample in range(num_samples):
                    current_loss, _ = session.run([self.loss, self.training_operation],
                                                  feed_dict={self.x: [arg_data[sample]]})
                if index % interval == 0:
                    logger.debug('epoch: %d, loss = %.4f' % (index, current_loss))
                    self.saver.save(session, self.model_file)

    def test(self, arg_data):
        with tf.Session() as session:
            self.saver.restore(session, self.model_file)
            hidden, result = session.run([self.encoded, self.decoded], feed_dict={self.x: arg_data})
        logger.debug('input : %s' % arg_data)
        logger.debug('compressed : %s' % hidden)
        logger.debug('reconstructed : %s' % result)
        return result


def get_batch(arg_data, size):
    section = np.random.choice(len(arg_data), size, replace=False)
    result = arg_data[section]
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

    data = datasets.load_iris().data
    auto_encoder = Autoencoder(input_dim=data.shape[1], hidden_dim=1, epoch=500)
    auto_encoder.train(data, interval=25)

    test_data = [[8, 4, 6, 2]]
    auto_encoder.test(arg_data=test_data)

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
