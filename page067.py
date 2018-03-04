import logging
import time

import numpy as np
import tensorflow as tf

start_time = time.time()


def model(arg_x, arg_w, arg_num_coefficients):
    terms = [tf.multiply(tf.slice(arg_w, [local_index], [1]), tf.pow(arg_x, local_index)) for local_index in
             range(arg_num_coefficients)]
    result = tf.add_n(terms)
    return result


def split_dataset(arg_x, arg_y, arg_ratio):
    array = np.arange(x_dataset.size)
    np.random.shuffle(array)
    train_size = int(arg_ratio * arg_x.size)
    x_train_result = arg_x[array[0:train_size]]
    x_test_result = arg_x[array[train_size:arg_x.size]]
    y_train_result = arg_y[array[0:train_size]]
    y_test_result = arg_y[array[train_size:arg_x.size]]
    return x_train_result, x_test_result, y_train_result, y_test_result


formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

random_seed = 67
np.random.seed(random_seed)
learning_rate = 0.001
regularization_lambda = 0.0

x_dataset = np.linspace(-1.0, 1.0, 100)

coefficients_count = 9
y_dataset_parameters = [0.0] * coefficients_count
y_dataset_parameters[2] = 1.0
y_dataset = 0
for index in range(coefficients_count):
    y_dataset += y_dataset_parameters[index] * np.power(x_dataset, index)
y_dataset += np.random.rand(*(len(x_dataset),)) * 0.3

split_ratio = 0.7
(x_train, x_test, y_train, y_test) = split_dataset(x_dataset, y_dataset, split_ratio)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

name = 'parameters'
w = tf.Variable([0.0] * coefficients_count, name=name)
y_model = model(X, w, coefficients_count)
cost = tf.div(tf.add(tf.reduce_sum(tf.square(Y - y_model)),
                     tf.multiply(regularization_lambda, tf.reduce_sum(tf.square(w)))),
              2.0 * x_train.size)

training_operation = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

session = tf.Session()
initializer = tf.global_variables_initializer()
session.run(initializer)

training_epochs = 1000

for regularization_lambda in np.linspace(0, 1, 100):
    for epoch in range(training_epochs):
        feed_dict_train = {X: x_train, Y: y_train}
        session.run(training_operation, feed_dict=feed_dict_train)
    feed_dict_test = {X: x_test, Y: y_test}
    final_cost = session.run(cost, feed_dict=feed_dict_test) * 100.0
    logger.debug('lambda: %.2f' % regularization_lambda)
    logger.debug('final cost: %.4f' % final_cost)

session.close()

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
