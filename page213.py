import logging
import time

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq

import page206


def make_cell(state_dimension):
    lstm_initializer = tf.random_uniform_initializer(-0.1, 0.1)
    result = rnn.LSTMCell(state_dimension, initializer=lstm_initializer)
    return result


def make_multi_cell(state_dimension, number_of_layers):
    cells = [make_cell(state_dimension=state_dimension) for each in range(number_of_layers)]
    result = rnn.MultiRNNCell(cells)
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

    input_sentences = ['hello stranger', 'bye bye']
    output_sentences = ['hiya', 'later alligator']
    input_int_to_symbol, input_symbol_to_int = page206.extract_character_vocabulary(input_sentences)
    output_int_to_symbol, output_symbol_to_int = page206.extract_character_vocabulary(output_sentences)

    logger.debug(input_int_to_symbol)
    logger.debug(input_symbol_to_int)
    logger.debug(output_int_to_symbol)
    logger.debug(output_symbol_to_int)

    epoch_count = 300
    rnn_state_dimension = 512
    rnn_layer_count = 2
    encoder_embedding_dimension = 64
    decoder_embedding_dimension = 64
    batch_size = int(32)
    learning_rate = 3e-4
    input_vocabulary_size = len(input_symbol_to_int)
    output_vocabulary_size = len(output_symbol_to_int)

    encoder_input_sequence = tf.placeholder(tf.int32, [None, None], name='encoder_input_sequence')
    encoder_sequence_length = tf.placeholder(tf.int32, (None,), name='encoder_sequence_length')
    decoder_output_sequence = tf.placeholder(tf.int32, [None, None], name='encoder_output_sequence')
    decoder_sequence_length = tf.placeholder(tf.int32, (None,), name='decoder_sequence_length')
    max_decoder_sequence_length = tf.reduce_max(decoder_sequence_length, name='max_decoder_sequence_length')

    encoder_input_embedded = layers.embed_sequence(encoder_input_sequence, input_vocabulary_size,
                                                   encoder_embedding_dimension)
    encoder_multi_cell = make_multi_cell(rnn_state_dimension, rnn_layer_count)
    encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_multi_cell, encoder_input_embedded,
                                                      sequence_length=encoder_sequence_length, dtype=tf.float32)

    del encoder_output

    decoder_raw_sequence = decoder_output_sequence[:, :-1]
    go_prefixes = tf.fill([batch_size, 1], output_symbol_to_int['<GO>'])
    decoder_input_sequence = tf.concat([go_prefixes, decoder_raw_sequence], 1)

    decoder_embedding = tf.Variable(tf.random_uniform([output_vocabulary_size, decoder_embedding_dimension]))
    decoder_input_embedded = tf.nn.embedding_lookup(decoder_embedding, decoder_input_sequence)
    decoder_multi_cell = make_multi_cell(rnn_state_dimension, rnn_layer_count)
    output_layer_kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1.0)
    output_layer = tf.layers.Dense(output_vocabulary_size, kernel_initializer=output_layer_kernel_initializer)

    with tf.variable_scope('decode'):
        training_helper = seq2seq.TrainingHelper(inputs=decoder_input_embedded,
                                                 sequence_length=decoder_sequence_length, time_major=False)
        training_decoder = seq2seq.BasicDecoder(decoder_multi_cell, training_helper, encoder_state, output_layer)
        training_decoder_output_sequence, _, _ = seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                                        maximum_iterations=max_decoder_sequence_length)

    with tf.variable_scope('decode', reuse=True):
        start_tokens = tf.tile(tf.constant([output_symbol_to_int['<GO>']], dtype=tf.int32), [batch_size],
                               name='start_tokens')
        inference_helper = seq2seq.GreedyEmbeddingHelper(embedding=decoder_embedding, start_tokens=start_tokens,
                                                         end_token=output_symbol_to_int['<EOS>'])
        inference_decoder = seq2seq.BasicDecoder(decoder_multi_cell, inference_helper, encoder_state, output_layer)
        inference_decoder_output_sequence, _, _ = seq2seq.dynamic_decode(inference_decoder, impute_finished=True,
                                                                         maximum_iterations=max_decoder_sequence_length)

    training_logits = tf.identity(training_decoder_output_sequence.rnn_output, name='logits')
    inference_logits = tf.identity(inference_decoder_output_sequence.sample_id, name='predictions')
    masks = tf.sequence_mask(decoder_sequence_length, max_decoder_sequence_length, dtype=tf.float32, name='masks')
    cost = seq2seq.sequence_loss(training_logits, decoder_output_sequence, masks)



    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
