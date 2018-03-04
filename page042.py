import logging
import time

start_time = time.time()

import tensorflow as tf

# set up logging
formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

raw_data = [1.0, 2.0, 8.0, -1.0, 0.0, 5.5, 6.0, 13.0]

# first version as written in book
session = tf.InteractiveSession()
spike = tf.Variable(False)
spike.initializer.run()

for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i - 1] > 5:
        updater = tf.assign(spike, True)
        updater.eval()
    else:
        tf.assign(spike, False).eval()
    logger.debug('spike: %s' % spike.eval())

session.close()

# second version with no updater, code in line
session = tf.InteractiveSession()
spike = tf.Variable(False)
spike.initializer.run()

for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i - 1] > 5:
        tf.assign(spike, True).eval()
    else:
        tf.assign(spike, False).eval()
    logger.debug('spike: %s' % spike.eval())
session.close()

# third version with no conditional
session = tf.InteractiveSession()
spike = tf.Variable(False)
spike.initializer.run()

for i in range(1, len(raw_data)):
    result = raw_data[i] - raw_data[i - 1] > 5
    tf.assign(spike, result).eval()
    logger.debug('spike: %s' % spike.eval())
session.close()

# fourth version with no temporary
session = tf.InteractiveSession()
spike = tf.Variable(False)
spike.initializer.run()

for i in range(1, len(raw_data)):
    tf.assign(spike, raw_data[i] - raw_data[i - 1] > 5).eval()
    logger.debug('spike: %s' % spike.eval())
session.close()


logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
