import logging
import time

import matplotlib.pyplot as plt
import numpy as np


def get_data(arg_feature_count, arg_output_file=None):
    result_a = np.random.rand(10, arg_feature_count) + 1
    result_b = np.random.rand(10, arg_feature_count)
    plt.scatter(result_a[:, 0], result_a[:, 1], c='r', marker='x')
    plt.scatter(result_b[:, 0], result_b[:, 1], c='g', marker='o')
    if arg_output_file is None:
        plt.show()
    else:
        plt.savefig(arg_output_file)
    return result_a, result_b

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

    feature_count = 2
    random_seed = 1
    np.random.seed(random_seed)
    scatter_plot_file = './output/page227-scatter.png'
    logger.debug('writing scatter plot to %s' % scatter_plot_file)
    data_a, data_b = get_data(arg_feature_count=feature_count, arg_output_file=scatter_plot_file)

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
