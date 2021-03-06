import logging
import time

import tensorflow as tf
import tensorflow.contrib.rnn as rnn


def make_cell(state_dimension):
    result = rnn.LSTMCell(state_dimension)
    return result


def make_multi_cell(state_dimension, number_of_layers):
    cells = [make_cell(state_dimension=state_dimension) for each in range(number_of_layers)]
    result = rnn.MultiRNNCell(cells)
    return result


def extract_character_vocabulary(data):
    special_symbols = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    set_symbols = set([character for line in data for character in line])
    all_symbols = special_symbols + list(set_symbols)
    int_to_symbol = {word_i: word for word_i, word in enumerate(all_symbols)}
    symbol_to_int = {word: word_i for word_i, word in enumerate(all_symbols)}
    return int_to_symbol, symbol_to_int

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

    input_dimension = 1
    sequence_size = 6
    shape = [None, sequence_size, input_dimension]
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=shape)

    with tf.variable_scope('first_cell') as scope_1:
        cell_1 = make_cell(state_dimension=10)
        outputs_1, states_1 = tf.nn.dynamic_rnn(cell_1, input_placeholder, dtype=tf.float32)

    with tf.variable_scope('second_cell') as scope_2:
        cell_2 = make_cell(state_dimension=10)
        outputs_2, states_2 = tf.nn.dynamic_rnn(cell_2, outputs_1, dtype=tf.float32)

    multi_cell = make_multi_cell(state_dimension=10, number_of_layers=4)
    outputs_4, states_4 = tf.nn.dynamic_rnn(multi_cell, input_placeholder, dtype=tf.float32)

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
