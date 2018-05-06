import logging
import time

import tensorflow as tf

from page174a import read_data

start_time = time.time()


# todo try rolling this up into one line
def conv_layer(arg_x, arg_W, arg_b):
    conv = tf.nn.conv2d(arg_x, arg_W, strides=[1, 1, 1, 1], padding='SAME')
    conv_with_b = tf.nn.bias_add(conv, arg_b)
    result = tf.nn.relu(conv_with_b)
    return result


def maxpool_layer(arg_conv, arg_k=2):
    shape = [1, arg_k, arg_k, 1]
    result = tf.nn.max_pool(arg_conv, ksize=shape, strides=shape, padding='SAME')
    return result


# todo give this function some arguments so we are not grabbing them from file scope
def model():
    x_reshaped = tf.reshape(x, shape=[-1, 24, 24, 1])
    conv_out1 = conv_layer(x_reshaped, W1, b1)
    maxpool_out1 = maxpool_layer(conv_out1)
    norm1 = tf.nn.lrn(maxpool_out1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    conv_out2 = conv_layer(norm1, W2, b2)
    norm2 = tf.nn.lrn(conv_out2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    maxpool_out2 = maxpool_layer(norm2)
    maxpool_reshaped = tf.reshape(maxpool_out2, [-1, W3.get_shape().as_list()[0]])
    local = tf.add(tf.matmul(maxpool_reshaped, W3), b3)
    local_out = tf.nn.relu(local)
    result = tf.add(tf.matmul(local_out, W_out), b_out)
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

    names, data, labels = read_data('./cifar-10-batches-py', arg_logger=logger)
    x = tf.placeholder(tf.float32, (None, 2 * 24))
    y = tf.placeholder(tf.float32, (None, len(names)))
    W1 = tf.Variable(tf.random_normal([5, 5, 1, 64]))
    b1 = tf.Variable(tf.random_normal([64]))
    W2 = tf.Variable(tf.random_normal([5, 5, 64, 64]))
    b2 = tf.Variable(tf.random_normal([64]))
    W3 = tf.Variable(tf.random_normal([6 * 6 * 64, 1024]))
    b3 = tf.Variable(tf.random_normal([1024]))
    W_out = tf.Variable(tf.random_normal([1024, len(names)]))
    b_out = tf.Variable(tf.random_normal([len(names)]))
    model_op = model()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_op, labels=y))
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(model_op, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        onehot_labels = tf.one_hot(labels, len(names), on_value=1.0, off_value=0.0, axis=-1)


    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
