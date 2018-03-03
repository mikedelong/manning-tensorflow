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

random_seed = 92
np.random.seed(random_seed)
x1_label0 = np.random.normal(1, 1, (100, 1))
x2_label0 = np.random.normal(1, 1, (100, 1))
x1_label1 = np.random.normal(5, 1, (100, 1))
x2_label1 = np.random.normal(4, 1, (100, 1))
x1_label2 = np.random.normal(8, 1, (100, 1))
x2_label2 = np.random.normal(0, 1, (100, 1))

plt.scatter(x1_label0, x2_label0, c='r', marker='o', s=60)
plt.scatter(x1_label1, x2_label1, c='g', marker='x', s=60)
plt.scatter(x1_label2, x2_label2, c='b', marker='_', s=60)
output_file = './output/scatter92.png'
plt.savefig(output_file)

xs_label0 = np.hstack((x1_label0, x2_label0))
xs_label1 = np.hstack((x1_label1, x2_label1))
xs_label2 = np.hstack((x1_label2, x2_label2))
xs = np.vstack((xs_label0, xs_label1, xs_label2))

labels = np.matrix(
    [[1.0, 0.0, 0.0]] * len(x1_label0) + [[0.0, 1.0, 0.0]] * len(x1_label1) + [[0.0, 0.0, 1.0]] * len(x1_label2))

arrangement = np.arange(xs.shape[0])
np.random.shuffle(arrangement)
xs = xs[arrangement, :]
labels = labels[arrangement, :]

test_x1_label0 = np.random.normal(1, 1, (10, 1))
test_x2_label0 = np.random.normal(1, 1, (10, 1))
test_x1_label1 = np.random.normal(5, 1, (10, 1))
test_x2_label1 = np.random.normal(4, 1, (10, 1))
test_x1_label2 = np.random.normal(8, 1, (10, 1))
test_x2_label2 = np.random.normal(0, 1, (10, 1))
test_xs_label0 = np.hstack((test_x1_label0, test_x2_label0))
test_xs_label1 = np.hstack((test_x1_label1, test_x2_label1))
test_xs_label2 = np.hstack((test_x1_label2, test_x2_label2))

test_xs = np.vstack((test_xs_label0, test_xs_label1, test_xs_label2))
test_labels = np.matrix(
    [[1.0, 0.0, 0.0]] * 10 + [[0.0, 1.0, 0.0]] * 10 + [[0.0, 0.0, 1.0]] * 10)

train_size, feature_count = xs.shape

epoch_count = 1000
label_count = 3
batch_size = 100

X = tf.placeholder("float", shape=[None, feature_count])
Y = tf.placeholder("float", shape=[None, label_count])
W = tf.Variable(tf.zeros([feature_count, label_count]))
b = tf.Variable(tf.zeros([label_count]))
y_model = tf.nn.softmax(tf.matmul(X, W) + b)

cost = -tf.reduce_sum(Y * tf.log(y_model))
learning_rate = 0.01
training_operation = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

with tf.Session() as session:
    tf.global_variables_initializer().run()

    for step in range(epoch_count * train_size // batch_size):
        offset = (step * batch_size) % train_size
        batch_xs = xs[offset: (offset + batch_size), :]
        batch_labels = labels[offset: (offset + batch_size)]
        feed_dict = {X: batch_xs, Y: batch_labels}
        error, _ = session.run([cost, training_operation], feed_dict=feed_dict)
        logger.debug('at step %d error is %.4f' % (step, error))

    w_result = session.run(W)
    logger.debug('w: %s' % w_result)
    b_result = session.run(b)
    logger.debug('b: %s' % b_result)
    logger.debug('accuracy: %.4f' % accuracy.eval(feed_dict={X: test_xs, Y: test_labels}))


logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
