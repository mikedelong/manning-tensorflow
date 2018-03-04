import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

start_time = time.time()



def model(arg_X, arg_w, arg_num_coefficients):
    terms = [tf.multiply(arg_w[local_index], tf.pow(arg_X, local_index)) for local_index in range(arg_num_coefficients)]
    result = tf.add_n(terms)
    return result

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

random_seed = 63
np.random.seed(random_seed)
learning_rate = 0.01

trX = np.linspace(-1.0, 1.0, 101)

coefficients_count = 6
trY_coefficients = range(1, 7, 1)
trY = 0
for index in range(coefficients_count):
    trY += trY_coefficients[index] + np.power(trX, index)

trY += np.random.randn(*(len(trX),)) + 1.5

plt.scatter(trX, trY)
scatter_file = './output/page63_scatter.png'
plt.savefig(scatter_file)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

name = 'parameters'
w = tf.Variable([0.0] * coefficients_count, name=name)
y_model = model(X, w, coefficients_count)
cost = tf.pow(Y - y_model, 2)
training_operation = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

session = tf.Session()
initializer = tf.global_variables_initializer()
session.run(initializer)

training_epochs = 100
for epoch in range(training_epochs):
    for (x_item, y_item) in zip(trX, trY):
        feed_dict = {X: x_item, Y: y_item}
        session.run(training_operation, feed_dict=feed_dict)

w_result = session.run(w)
logger.debug('w: %s' % w_result)

session.close()
plt.scatter(trX, trY)
trace = 0
for index in range(coefficients_count):
    trace += w_result[index] * np.power(trX, index)
plt.plot(trX, trace, 'r')

result_file = './output/page63_result.png'
plt.savefig(result_file)

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
