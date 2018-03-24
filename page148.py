import logging
import pickle
import time

import numpy as np

start_time = time.time()


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

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
