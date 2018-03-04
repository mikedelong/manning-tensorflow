import logging
import time

import bregman
import numpy as np
import tensorflow as tf

start_time = time.time()


def get_next_chromagram(arg_session, arg_filename):
    local_audio_file = arg_session.run(arg_filename)
    result = bregman.suite.Chromagram(local_audio_file, nfft=16384, wfft=8192, nhop=2205)
    return result.X


def extract_feature_vector(arg_session, arg_chroma_data, arg_max_frequencies):
    feature_count, sample_count = np.shape(arg_chroma_data)
    frequency_values = arg_session.run(arg_max_frequencies, feed_dict={chroma: arg_chroma_data})
    histogram, _ = np.histogram(frequency_values, bins=range(feature_count + 1))
    result = histogram.astype(float) / sample_count
    return result

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

filenames = tf.train.match_filenames_once('./input/audio_dataset/*.wav')
filename_count = tf.size(filenames)
filename_queue = tf.train.string_input_producer(filenames)
reader = tf.WholeFileReader()
file_name, file_contents = reader.read(filename_queue)

chroma = tf.placeholder(tf.float32)
max_frequencies = tf.argmax(chroma, 0)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    file_count = session.run(filename_count)
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coordinator)

    for index in range(file_count):
        audio_file = session.run(file_name)
        logger.debug('file name: %s' % audio_file)

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
