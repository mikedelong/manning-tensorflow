import logging
import time

import tensorflow as tf

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

    def train(self, data):
        num_samples = len(data)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for index in range(self.epoch):
                l = None
                for sample in range(num_samples):
                    l, _ = session.run([self.loss, self.training_operation], feed_dict={self.x: [data[sample]]})
                if index % 10 == 0:
                    logger.debug('epoch: %d, loss = %.4f' % (index, l))
                    self.saver.save(session, self.model_file)

    def test(self, data):
        with tf.Session() as session:
            self.saver.restore(session, self.model_file)
            hidden, reconstructed = session.run([self.encoded, self.decoded], feed_dict={self.x: data})
        logger.debug('input : %s' % data)
        logger.debug('compressed : %s' % hidden)
        logger.debug('reconstructed : %s' % reconstructed)
        return reconstructed



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
