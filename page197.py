import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from page195 import SeriesPredictor
from page196 import load_series
from page196 import split_data


def plot_results(train_x, predictions, actual, filename):
    plt.figure()
    num_train = len(train_x)
    plt.plot(list(range(num_train)), train_x, color='b', label='training data')
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='predicted')
    plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    start_time = time.time()

    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    sequence_size = 5
    predictor = SeriesPredictor(input_dim=1, seq_size=sequence_size, hidden_dim=100)
    data = load_series('./international-airline-passengers.csv', series_idx=1, arg_logger=logger)
    train_data, actual_vals = split_data(data)

    train_x = [np.expand_dims(train_data[i:i + sequence_size], axis=1).tolist() for i in
               range(len(train_data) - sequence_size - 1)]
    train_y = [train_data[i + 1:i + sequence_size + 1] for i in range(len(train_data) - sequence_size - 1)]
    test_x = [np.expand_dims(actual_vals[i:i + sequence_size], axis=1).tolist() for i in
              range(len(actual_vals) - sequence_size - 1)]
    test_y = [actual_vals[i + 1:i + sequence_size + 1] for i in range(len(actual_vals) - sequence_size - 1)]

    predictor.train(train_x=train_x, train_y=train_y)

    with tf.Session() as session:
        predicted_values = predictor.test(test_x=test_x, arg_session=session)[:, 0]
        logger.debug('predicted values shape: %s' % np.shape(predicted_values))
        plot_results(train_data, predicted_values, actual_vals, './output/page197-predictions.png')
        previous_sequence = train_x[-1]
        predicted_values = list()
        for i in range(20):
            next_sequence = predictor.test(session, [previous_sequence])
            predicted_values.append(next_sequence[-1])
            previous_sequence = np.vstack((previous_sequence[1:], next_sequence[-1]))
            plot_results(train_data, predicted_values, actual_vals, './output/page197-hallucinations.png')

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
