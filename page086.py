import logging
import time

import matplotlib.pyplot as plt
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

random_seed = 86
x1 = np.random.normal(-4, 2, 1000)
x2 = np.random.normal(4, 2, 1000)
xs = np.append(x1, x2)
ys = np.asarray([0.0] * len(x1) + [1.0] * len(x2))
plt.scatter(xs, ys)
scatter_file = './output/page86_scatter.png'
plt.savefig(scatter_file)

shape = (None,)
X = tf.placeholder(tf.float32, shape=shape, name='x')
Y = tf.placeholder(tf.float32, shape=shape, name='y')
w = tf.Variable([0.0, 0.0], name='parameter', trainable=True)
y_model = tf.sigmoid(tf.slice(w, [1], [1]) * X + tf.slice(w, [0], [1]))
cost = tf.reduce_mean(-Y * tf.log(y_model) - (1 - Y) * tf.log(1 - y_model))
learning_rate = 0.01
training_operation = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

epoch_count = 1000
tolerance = 0.0001
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    previous_error = 0
    feed_dict = {X: xs, Y: ys}
    for epoch in range(epoch_count):
        error, _ = session.run([cost, training_operation], feed_dict=feed_dict)
        logger.debug('epoch: %d error: %.4f' % (epoch, error))
        if abs(previous_error - error) < tolerance:
            break
        previous_error = error
    w_result = session.run(w, feed_dict=feed_dict)

all_xs = np.linspace(-10, 10, 100)
plt.plot(all_xs, sigmoid((all_xs * w_result[1] + w_result[0])))
result_file = './output/page86_result.png'
plt.savefig(result_file)


logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
