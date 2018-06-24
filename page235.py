import glob
import logging
import os
import time

import numpy as np
import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imresize

from vgg16 import vgg16

DATASET_DIR = os.path.join(os.path.expanduser('-'), 'res', 'cloth_folding_rgb_vids')
NUM_VIDS = 45


def get_image_pair(video_id):
    image_files = sorted(glob.glob(os.path.join(DATASET_DIR, video_id, '*.png')))
    first_image = image_files[0]
    last_image = image_files[-1]
    pair = []
    for image_file in [first_image, last_image]:
        image_original = imread(image_file)
        image_resized = imresize(image_original, [224, 224])
        pair.append(image_resized)
    result = tuple(pair)
    return result


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

    start_images = []
    end_images = []
    for vid_id in range(1, NUM_VIDS + 1):
        start_image, end_image = get_image_pair(str(vid_id))
        start_images.append(start_image)
        end_images.append(end_image)
    logger.debug('Images of starting state {}'.format(np.shape(start_images)))
    logger.debug('Images of ending state {}'.format(np.shape(end_images)))

    images_placeholder = tf.placeholder(tf.float32, [None, 224, 224, 3])
    feature_count = 2
    hidden_count = 10

    with tf.name_scope('input'):
        x1 = tf.placeholder(tf.float32, [None, feature_count], name='x1')
        x2 = tf.placeholder(tf.float32, [None, feature_count], name='x2')
        dropout_keep_probability = tf.placeholder(tf.float32, name='dropout_probability')

    with tf.name_scope('hidden_layer'):
        with tf.name_scope('weights'):
            w1 = tf.Variable(tf.random_normal([feature_count, hidden_count]), name='w1')
            tf.summary.histogram('w1', w1)
            b1 = tf.Variable(tf.random_normal([hidden_count]), name='b1')
            tf.summary.histogram('b1', b1)

        with tf.name_scope('output'):
            h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x1, w1) + b1), keep_prob=dropout_keep_probability)
            tf.summary.histogram('h1', h1)
            h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(x2, w1) + b1), keep_prob=dropout_keep_probability)
            tf.summary.histogram('h2', h2)

    with tf.name_scope('output_layer'):
        with tf.name_scope('output_layer'):
            w2 = tf.Variable(tf.random_normal([hidden_count, 1]), name='w2')
            tf.summary.histogram('w2', w2)
            b2 = tf.Variable(tf.random_normal([1]), name='b2')
            tf.summary.histogram('b2', b2)
        with tf.name_scope('output'):
            s1 = tf.matmul(h1, w2) + b2
            s2 = tf.matmul(h2, w2) + b2

    with tf.name_scope('loss'):
        s12 = s1 - s2
        s12_flat = tf.reshape(s12, [-1])
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.zeros_like(s12_flat), logits=s12_flat + 1)
        loss = tf.reduce_mean(cross_entropy)

    with tf.name_scope('train_op'):
        train_operation = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    logger.debug('loading VGG16 model')
    vgg = vgg16(images_placeholder, 'vgg16_weights.npz', session)
    logger.debug('done loading the VGG16 model.')

    start_images_embedded = session.run(vgg.fc1, feed_dict={vgg.imgs: start_images})
    end_images_embedded = session.run(vgg.fc1, feed_dict={vgg.imgs: end_images})
    indexes = np.random.choice(NUM_VIDS, NUM_VIDS, replace=False)
    train_indexes = indexes[0:int(NUM_VIDS * 0.75)]
    test_indexes = indexes[int(NUM_VIDS * 0.75):]

    train_start_images = start_images_embedded[train_indexes]
    train_end_images = end_images_embedded[train_indexes]
    test_start_images = start_images_embedded[test_indexes]
    test_end_images = end_images_embedded[test_indexes]

    logger.debug('Train start images {}'.format(np.shape(train_start_images)))
    logger.debug('Train end images {}'.format(np.shape(train_end_images)))
    logger.debug('Test start images {}'.format(np.shape(test_start_images)))
    logger.debug('Test end images {}'.format(np.shape(test_end_images)))

    train_y1 = np.expand_dims(np.zeros(np.shape(train_start_images)[0]), axis=1)
    train_y2 = np.expand_dims(np.ones(np.shape(train_end_images)[0]), axis=1)
    for epoch in range(100):
        for index in range(np.shape(train_start_images)[0]):
            _, cost_value = session.run([train_operation, loss], feed_dict={x1: train_start_images[index:index + 1, :],
                                                                            x2: train_end_images[index:index + 1, :],
                                                                            dropout_keep_probability: 0.5})
            logger.debug('{} {}'.format(epoch, cost_value))
            s1_value, s2_value = session.run([s1, s2], feed_dict={x1: test_start_images,
                                                                  x2: test_end_images,
                                                                  dropout_keep_probability: 1.0})
            logger.debug('Accuracy: {}%'.format(100 * np.mean(s1_value < s2_value)))

    # todo move this out of __main__
    def get_image_sequence(video_id):
        image_files = sorted(glob.glob(os.path.join(DATASET_DIR, video_id, '*.png')))
        result = list()
        # todo use a comprehension?
        for image_file in image_files:
            image_original = imread(image_file)
            image_resized = imresize(image_original, (224, 224))
            result.append(image_resized)
        return result


    images = get_image_sequence('1')
    images_embedded = session.run(vgg.fc1, feed_dict={vgg.imgs: images})
    scores = session.run([s1], feed_dict={x1: images_embedded, dropout_keep_probability: 1.0})

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
