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

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
