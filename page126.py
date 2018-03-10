import logging
import time

import numpy as np
import tensorflow as tf

start_time = time.time()


class HMM(object):
    def __init__(self, initial_prob, trans_prob, obs_prob):
        self.N = np.size(initial_prob)
        self.initial_prob = initial_prob
        self.trans_prob = trans_prob
        self.emission = tf.constant(obs_prob)

        assert self.initial_prob.shape == (self.N, 1)
        assert self.trans_prob.shape == (self.N, self.N)
        assert obs_prob.shape[0] == self.N

        self.obs_idx = tf.placeholder(tf.int32)
        self.fwd = tf.placeholder(tf.float64)

    def get_emission(self, obs_idx):
        slice_location = [0, obs_idx]
        num_rows = tf.shape(self.emission)[0]
        slice_shape = [num_rows, 1]
        result = tf.slice(self.emission, slice_location, slice_shape)
        return result

    def forward_init_op(self):
        obs_prob = self.get_emission(self.obs_idx)
        result = tf.multiply(self.initial_prob, obs_prob)
        return result

    def forward_op(self):
        transitions = tf.matmul(self.fwd, tf.transpose(self.get_emission(self.obs_idx)))
        weighted_transitions = transitions * self.trans_prob
        fwd = tf.reduce_sum(weighted_transitions, 0)
        result = tf.reshape(fwd, tf.shape(self.fwd))
        return result


def forward_algorithm(session, model, observations):
    fwd = session.run(model.forward_init_op(), feed_dict={model.obs_idx: observations[0]})
    for t in range(1, len(observations)):
        fwd = session.run(model.forward_op(), feed_dict={model.obs_idx: observations[t], model.fwd: fwd})
        result = session.run(tf.reduce_sum(fwd))
        return result


if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    initial_prob = np.array([[0.6], [0.4]])
    trans_prob = np.array([[0.7, 0.3], [0.4, 0.6]])
    obs_prob = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])

    model = HMM(initial_prob=initial_prob, trans_prob=trans_prob, obs_prob=obs_prob)
    observations = [0, 1, 1, 2, 1]
    with tf.Session() as session:
        prob = forward_algorithm(session, model, observations)
        logger.debug('probability of observing %s is %.4f' % (observations, prob))

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
