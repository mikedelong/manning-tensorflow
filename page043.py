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

session = tf.InteractiveSession()
spikes = tf.Variable([False] * len(raw_data), name='spikes')
spikes.initializer.run()
saver = tf.train.Saver()

for i in range(1, len(raw_data)):
    if (raw_data[i] - raw_data[i - 1] > 5):
        spikes_val = spikes.eval()
        spikes_val[i] = True
        updater = tf.assign(spikes, spikes_val)
        updater.eval()

checkpoint_filename = './spikes.ckpt'
save_path = saver.save(session, checkpoint_filename)
logger.debug('spikes data saved to %s' % save_path)

session.close()

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
