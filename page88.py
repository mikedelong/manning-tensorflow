import logging
import time

import numpy as np

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
learning_rate = 0.1
epoch_count = 2000

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

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
