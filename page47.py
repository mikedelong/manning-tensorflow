import logging
import time

import numpy as np
import tensorflow as tf

start_time = time.time()

# set up logging
formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

random_seed = 47
np.random.seed(seed=random_seed)
raw_data = np.random.normal(10, 1, 100)
alpha = tf.constant(0.05)
beta = tf.Variable(1.0 - alpha)
current_value = tf.placeholder(tf.float32)
previous_average = tf.Variable(0.0)
update_average = alpha * current_value + tf.multiply(beta, previous_average)

average_history = tf.summary.scalar('running average', update_average)
value_history = tf.summary.scalar('incoming values', current_value)
merged = tf.summary.merge_all()
log_folder = './logs/'
writer = tf.summary.FileWriter(log_folder)

initializer = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(initializer)
    writer.add_graph(session.graph)
    for index, value in enumerate(raw_data):
        feed_dict = {current_value: value}
        summary_string, current_average = session.run([merged, update_average], feed_dict=feed_dict)
        session.run(tf.assign(previous_average, current_average))
        logger.debug('raw data: %.2f current average: %.2f' % (value, current_average))
        writer.add_summary(summary_string, index)

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
