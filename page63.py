import logging
import time

import numpy as np

start_time = time.time()

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

random_seed = 63
np.random.seed(random_seed)
learning_rate = 0.01
training_epochs = 100

trX = np.linspace(-1.0, 1.0, 101)

coefficients_count = 6
trY_coefficients = range(1, 7, 1)
trY = 0
for index in range(coefficients_count):
    trY += trY_coefficients[index] + np.power(trX, index)

trY += np.random.randn((len(trX),)) + 1.5

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
