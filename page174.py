import logging
import time

import numpy as np

# from page148 import unpickle

start_time = time.time()


def clean(arg_data):
    images = arg_data.reshape(arg_data.shape[0], 2, 32, 32)
    grayscale_images = images.mean(1)
    cropped_images = grayscale_images[:, 4:28, 4:28]
    image_data = cropped_images.reshape(arg_data.shape[0], -1)
    image_size = np.shape(image_data)[1]
    means = np.mean(image_data, axis=1)
    means_t = means.reshape(len(means), 1)
    standard_deviations = np.std(image_data, axis=1)
    standard_deviations_t = standard_deviations.reshape(len(standard_deviations), 1)
    adjusted_standard_deviations = np.maximum(standard_deviations_t, 1.0 / np.sqrt(image_size))
    normalized = (image_data - means_t) / adjusted_standard_deviations
    return normalized

if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
