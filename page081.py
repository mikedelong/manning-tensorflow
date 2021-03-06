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


def model(arg_x, arg_w):
    result = tf.add(tf.multiply(tf.slice(arg_w, [1], [1]), tf.pow(arg_x, 1)),
                    tf.multiply(tf.slice(arg_w, [0], [1]), tf.pow(arg_x, 0)))
    return result


random_seed = 81
np.random.seed(random_seed)
x_label0 = np.random.normal(5, 1, 10)
x_label1 = np.random.normal(2, 1, 10)
xs = np.append(x_label0, x_label1)
labels = [0.0] * len(x_label0) + [1.0] * len(x_label1)
plt.scatter(xs, labels)
scatter_file = './output/page81_scatter.png'
plt.savefig(scatter_file)

learning_rate = 0.001
training_epochs = 1000

X = tf.placeholder("float")
Y = tf.placeholder("float")

name = 'parameters'
w = tf.Variable([0.0, 0.0], name=name)
y_model = model(X, w)
cost = tf.reduce_sum(tf.square(Y - y_model))
training_operation = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(Y, tf.to_float(tf.greater(y_model, 0.5)))
accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

session = tf.Session()
initializer = tf.global_variables_initializer()
session.run(initializer)

feed_dict = {X: xs, Y: labels}
for epoch in range(training_epochs):
    session.run(training_operation, feed_dict=feed_dict)
    current_cost = session.run(cost, feed_dict=feed_dict)
    if epoch % 100 == 0:
        logger.debug('epoch : %d, cost : %.4f' % (epoch, current_cost))
w_result = session.run(w)
logger.debug('learned parameters: %s' % w_result)
logger.debug('accuracy: %.3f' % session.run(accuracy, feed_dict=feed_dict))
session.close()

all_xs = np.linspace(0, 10, 100)
plt.plot(all_xs, all_xs * w_result[1] + w_result[0])
output_file = './output/page81.png'
plt.savefig(output_file)

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
