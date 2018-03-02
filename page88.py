import logging
import time

import numpy as np
import tensorflow as tf

start_time = time.time()


def sigmoid(arg_x):
    result = 1.0 / (1.0 + np.exp(-arg_x))
    return result


formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

random_seed = 88
np.random.seed(random_seed)

x1_label1 = np.random.normal(3, 1, 1000)
x2_label1 = np.random.normal(2, 1, 1000)
x1_label2 = np.random.normal(7, 1, 1000)
x2_label2 = np.random.normal(6, 1, 1000)

x1s = np.append(x1_label1, x1_label2)
logger.debug('we have %d x1s' % x1s.size)
x2s = np.append(x2_label1, x2_label2)
logger.debug('we have %d x2s' % x2s.size)
ys = np.asarray([0.0] * x1_label1.size + [1.0] * x1_label2.size)
logger.debug('we have %d ys' % ys.size)

X1 = tf.placeholder(tf.float32, shape=(None,), name='x1')
X2 = tf.placeholder(tf.float32, shape=(None,), name='x2')
Y = tf.placeholder(tf.float32, shape=(None,), name='y')
w = tf.Variable([0.0, 0.0, 0.0], name='w', trainable=True)

y_model = tf.sigmoid(tf.slice(w, [2], [1]) * X2 + tf.slice(w, [1], [1]) * X1 + tf.slice(w, [0], [1]))
cost = tf.reduce_mean(-tf.log(y_model * Y + (1.0 - y_model) * (1.0 - Y)))
learning_rate = 0.1
training_operation = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

epoch_count = 2000
tolerance = 0.0001
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    previous_error = 0.0
    feed_dict = {X1: x1s, X2: x2s, Y: ys}
    for epoch in range(epoch_count):
        error, _ = session.run([cost, training_operation], feed_dict=feed_dict)
        logger.debug('epoch: %d error: %.4f' % (epoch, error))
        if abs(previous_error - error) < tolerance:
            break
        previous_error = error
    w_result = session.run(w, feed_dict=feed_dict)


logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
