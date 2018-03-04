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

random_seed = 60
np.random.seed(random_seed)
learning_rate = 0.01
training_epochs = 100

x_train = np.linspace(-1.0, 1.0, 101)
y_train = np.multiply(2.0, x_train) + np.random.randn((len(x_train, ))) * 0.33
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
name = 'weights'
w = tf.Variable(0.0, name=name)
y_model = tf.multiply(X, w)
# https://github.com/BinRoot/TensorFlow-Book/blob/master/ch03_regression/Concept01_linear_regression.ipynb
cost = tf.reduce_mean(tf.square(Y - y_model))
train_operation = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

session = tf.Session()
initializer = tf.global_variables_initializer()
session.run(initializer)
for epoch in range(training_epochs):
    for (x_item, y_item) in zip(x_train, y_train):
        feed_dict = {X: x_item, Y: y_item}
        session.run(train_operation, feed_dict=feed_dict)

w_result = session.run(w)
session.close()

y_learned = x_train * w_result
outfile = './output/page60.png'

plt.scatter(x_train, y_train)
plt.plot(x_train, y_learned, 'r')
plt.savefig(outfile)

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
