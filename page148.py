import logging
import pickle
import time

import numpy as np

from page145 import Autoencoder

start_time = time.time()


def grayscale(arg_image):
    result = arg_image.reshape(arg_image.shape[0], 3, 32, 32).mean(1).reshape(arg_image.shape[0], -1)
    return result


def unpickle(arg_file):
    with open(arg_file, 'rb') as file_pointer:
        result = pickle.load(file_pointer, encoding='latin1')
    return result


if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    input_folder = './cifar-10-batches-py/'
    input_file = input_folder + 'batches.meta'
    names = unpickle(input_file)['label_names']
    data = list()
    labels = list()
    for i in range(1, 6):
        file_name = input_folder + 'data_batch_' + str(i)
        logger.debug('processing %s' % file_name)
        batch_data = unpickle(file_name)
        if len(data) > 0:
            data = np.vstack((data, batch_data['data']))
            labels = np.hstack((labels, batch_data['labels']))
        else:
            data = batch_data['data']
            labels = batch_data['labels']

    logger.debug('data length: %d, labels length: %d' % (len(data), len(labels)))
    data = grayscale(data)
    x = np.matrix(data)
    y = np.array(labels)
    horse_indices = np.where(y == 7)[0]
    horse_x = x[horse_indices]
    logger.debug('We expect the horse object to be 5000 x 3072 and it is %d x %d' % np.shape(horse_x))

    input_dim = np.shape(horse_x)[1]
    hidden_dim = 100
    auto_encoder = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    auto_encoder.train(horse_x, arg_logger=logger)

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
