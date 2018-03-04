import logging
import time

start_time = time.time()

import numpy as np
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

for matrix in [
    [[1.0, 2.0], [3.0, 4.0]],
    np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    tf.constant([[1.0, 2.0], [3.0, 4.0]])
]:
    logger.debug('matrix type: %s' % type(matrix))
    tensor = tf.convert_to_tensor(matrix)
    logger.debug('tensor type: %s' % type(tensor))

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
