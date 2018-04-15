import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from page148 import unpickle


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


def show_conv_results(arg_data, arg_file_name=None):
    plt.figure()
    rows = 4
    columns = 8
    size = np.shape(arg_data)[3]
    for index in range(size):
        image = arg_data[0, :, :, index]
        plt.subplot(rows, columns, index + 1)
        plt.imshow(image, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if arg_file_name:
        plt.savefig(arg_file_name)
    else:
        plt.show()


def show_weights(arg_data, filename=None):
    plt.figure()
    rows, cols = 4, 8
    for i in range(np.shape(arg_data)[3]):
        image = arg_data[:, :, 0, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

names, data, labels = read_data('./cifar-10-batches-py', arg_logger=logger)

raw_data = data[4, :]
raw_img = np.reshape(raw_data, (24, 24))
plt.figure()
plt.imshow(raw_img, cmap='Greys_r')

x = tf.reshape(raw_data, shape=[-1, 24, 24, 1])
W = tf.Variable(tf.random_normal([5, 5, 1, 32]))
b = tf.Variable(tf.random_normal([32]))

conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
conv_with_b = tf.nn.bias_add(conv, b)
conv_out = tf.nn.relu(conv_with_b)

k = 2
maxpool = tf.nn.max_pool(conv_out, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    W_val = sess.run(W)
    show_weights(W_val)

    conv_val = sess.run(conv)
    logger.debug('convolution results: %s' % str(np.shape(conv_val)))
    show_conv_results(conv_val)

    conv_out_val = sess.run(conv_out)
    logger.debug('convolution with bias and relu: %s ' % str(np.shape(conv_out_val)))
    show_conv_results(conv_out_val)

    maxpool_val = sess.run(maxpool)
    logger.debug('maxpool after all the convolutions: %s' % str(np.shape(maxpool_val)))
    show_conv_results(maxpool_val)
