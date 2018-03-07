import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

start_time = time.time()

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')


class SOM:
    def __init__(self, arg_width, arg_height, arg_dimensions):
        self.num_iters = 100
        self.width = arg_width
        self.height = arg_height
        self.dim = arg_dimensions
        self.node_locs = self.get_locs()

        nodes = tf.Variable(tf.random_normal([arg_width * arg_height, arg_dimensions]))
        self.nodes = nodes

        x = tf.placeholder(tf.float32, [arg_dimensions])
        _iter = tf.placeholder(tf.float32)
        self.x = x
        self.iter = _iter
        bmu_loc = self.get_bmu_loc(x)
        self.propagate_nodes = self.get_propagation(bmu_loc, x, _iter)

        self.nodes_val = None
        self.locs_val = None
        self.centroid_grid = None

    def get_propagation(self, bmu_loc, x, arg_iter):
        num_nodes = self.width * self.height
        rate = 1.0 - tf.div(arg_iter, self.num_iters)
        alpha = 0.5 * rate
        sigma = rate * tf.to_float(tf.maximum(self.width, self.height)) / 2.0
        expanded_bmu_loc = tf.expand_dims(tf.to_float(bmu_loc), 0)
        sqr_dists_from_bmu = tf.reduce_sum(tf.square(tf.subtract(expanded_bmu_loc, self.node_locs)), 1)
        neigh_factor = tf.exp(-tf.div(sqr_dists_from_bmu, 2 * tf.square(sigma)))
        rate = tf.multiply(alpha, neigh_factor)
        rate_factor = tf.stack([tf.tile(tf.slice(rate, [i], [1]), [self.dim]) for i in range(num_nodes)])
        nodes_diff = tf.multiply(rate_factor, tf.subtract(tf.stack([x for _ in range(num_nodes)]), self.nodes))
        update_nodes = tf.add(self.nodes, nodes_diff)
        result = tf.assign(self.nodes, update_nodes)
        return result

    def get_bmu_loc(self, x):
        expanded_x = tf.expand_dims(x, 0)
        sqr_diff = tf.square(tf.subtract(expanded_x, self.nodes))
        dists = tf.reduce_sum(sqr_diff, 1)
        bmu_idx = tf.argmin(dists, 0)
        result = tf.stack([tf.mod(bmu_idx, self.width), tf.div(bmu_idx, self.width)])
        return result

    def get_locs(self):
        locs = [(x, y) for y in range(self.height) for x in range(self.width)]
        result = tf.to_float(locs)
        return result

    def train(self, data):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for index in range(self.num_iters):
                for data_x in data:
                    session.run(self.propagate_nodes, feed_dict={self.x: data_x, self.iter: index})
            centroid_grid = [[] for _ in range(self.width)]
            self.nodes_val = list(session.run(self.nodes))
            self.locs_val = list(session.run(self.node_locs))
            for index, loc in enumerate(self.locs_val):
                centroid_grid[int(loc[0])].append(self.nodes_val[index])
            self.centroid_grid = centroid_grid


random_seed = 113
np.random.seed(random_seed)
tf.set_random_seed(random_seed)
colors = np.array(
    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.95], [0.0, 0.05, 1.0], [0.0, 1.0, 0.0], [0.0, 0.95, 0.0], [0.0, 1.0, 0.05],
     [1.0, 0.0, 0.0], [1.0, 0.05, 0.0], [1.0, 0.0, 0.05], [1.0, 1.0, 0.0]])

for width in range(4, 44, 4):
    for height in range(4, 44, 4):
        model = SOM(arg_width=width, arg_height=height, arg_dimensions=3)
        model.train(colors)
        plt.imshow(model.centroid_grid)
        output_file = './output/page113-w-' + str(width) + '-h-' + str(height) + '.png'
        logger.debug('writing case w: %d l: %d to %s' % (width, height, output_file))
        plt.savefig(output_file)

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
