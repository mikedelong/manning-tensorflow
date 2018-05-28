import logging
import time

import page206

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
    rnn_state_dimensions = 512
    rnn_layer_count = 2
    encoder_embedding_dimension = 64
    decoder_embedding_dimension = 64
    batch_size = int(32)
    learning_rate = 3e-4
    input_vocabulary_size = len(input_symbol_to_int)
    output_vocabulary_size = len(output_symbol_to_int)

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
