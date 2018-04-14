import logging
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from page148 import unpickle

start_time = time.time()


def clean(arg_data):
    images = arg_data.reshape(arg_data.shape[0], 3, 32, 32)
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


def read_data(arg_folder, arg_logger):
    result_names = unpickle('{}/batches.meta'.format(arg_folder))['label_names']
    arg_logger.debug('names: %s' % result_names)
    result_data = list()
    result_labels = list()
    for index in range(1, 6):
        file_name = '{}/data_batch_{}'.format(arg_folder, index)
        batch_data = unpickle(file_name)
        if len(result_data) > 0:
            result_data = np.vstack((result_data, batch_data['data']))
            result_labels = np.hstack((result_labels, batch_data['labels']))
        else:
            result_data = batch_data['data']
            result_labels = batch_data['labels']
    arg_logger.debug('data shape : %s labels shape : %s' % (np.shape(result_data), np.shape(result_labels)))
    result_data = clean(result_data)
    result_data = result_data.astype(np.float32)
    return result_names, result_data, result_labels


def show_some_examples(arg_names, arg_data, arg_labels, arg_output_file):
    plt.figure()
    rows = 4
    columns = 4
    random_indexes = random.sample(range(len(arg_data)), rows * columns)
    color_map = 'Greys_r'
    for index in range(rows * columns):
        plt.subplot(rows, columns, index + 1)
        j = random_indexes[index]
        plt.title(arg_names[arg_labels[j]])
        image = np.reshape(data[j, :], (24, 24))
        plt.imshow(image, cmap=color_map)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(arg_output_file)


def show_weights(arg_weights, arg_file_name=None):
    plt.figure()
    rows = 4
    columns = 8
    color_map = 'Greys_r'
    interpolation = 'none'
    for index in range(np.shape(arg_weights)[3]):
        image = arg_weights[:, :, 0, index]
        plt.subplot(rows, columns, index + 1)
        plt.imshow(image, cmap=color_map, interpolation=interpolation)
        plt.axis('off')
    if arg_file_name:
        plt.savefig(arg_file_name)
    else:
        plt.show()


def show_conv_results(arg_data, arg_file_name=None):
    plt.figure()
    rows = 4
    columns = 8
    for index in range(np.shape(arg_data)[3]):
        image = data[0, :, :, index]
        plt.subplot(rows, columns, index + 1)
        plt.imshow(image, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if arg_file_name:
        plt.savefig(arg_file_name)
    else:
        plt.show()

if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    names, data, labels = read_data('./cifar-10-batches-py', logger)
    output_folder = './output/'
    output_file = 'cifar_examples.png'
    full_output_file = output_folder + output_file
    logger.debug('saving some CIFAR example pictures to %s' % full_output_file)

    show_some_examples(arg_names=names, arg_data=data, arg_labels=labels, arg_output_file=full_output_file)

    weights = tf.Variable(tf.random_normal([5, 5, 1, 32]))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        weights_values = session.run(weights)
        show_weights(weights_values, './output/page174-step-0-weights.png')

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
