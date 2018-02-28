import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

start_time = time.time()

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')


def model(arg_X, arg_w):
    result = tf.add(tf.multiply(arg_w[1], tf.pow(arg_X, 1)), tf.multiply(arg_w[0], tf.pow(arg_X, 0)))
    return result

random_seed = 81
np.random.seed(random_seed)
x_label0 = np.random.normal(5, 1, 10)
x_label1 = np.random.normal(2, 1, 10)
xs = np.append(x_label0, x_label1)
labels = [0.0] * len(x_label0) + [1.0] * len(x_label1)
plt.scatter(xs, labels)

learning_rate = 0.001
training_epochs = 1000

X = tf.placeholder("float")
Y = tf.placeholder("float")

name = 'parameters'
w = tf.Variable([0.0, 0.0], name=name)
y_model = model(X, w)
cost = tf.reduce_sum(tf.square(Y - y_model))
training_operation = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))