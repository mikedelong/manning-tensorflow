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

session = tf.InteractiveSession()
spikes = tf.Variable([False] * 8, name='spikes')
spikes.initializer.run()
saver = tf.train.Saver()
checkpoint_filename = './spikes.ckpt'
saver.restore(session, checkpoint_filename)
logger.debug(spikes.eval())

session.close()

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
