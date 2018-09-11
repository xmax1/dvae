from __future__ import division, print_function

import numpy as np
import tensorflow as tf
import qupa
from qupa.pcd import PCD

tf.random.seed(0)

NUM_EPOCHS = 500
HIDDEN_SIZE = 500
BATCH_SIZE = 50
STATS_INTERVAL = 2

# training log Z parameters
NUM_SAMPLES = 128
NUM_MCMC_SWEEPS = 25

# evaluation log Z parameters
EVAL_NUM_SAMPLES = 2048
EVAL_NUM_STEPS = 200
EVAL_NUM_MCMC_SWEEPS = 25


def binarize(data):
    ones = np.random.random_sample(data.shape) < data
    return ones.astype(data.dtype)


def mean_free_energy(visible, visible_biases, hidden_biases, weights):
    with tf.name_scope('free_energy'):
        visible_term = tf.tensordot(visible, visible_biases, 1)
        hidden_terms = tf.reduce_sum(tf.nn.softplus(
            tf.matmul(visible, weights) + hidden_biases), 1)
        return -tf.reduce_mean(visible_term + hidden_terms)


def make_image(data):
    gx, gy = 10, 10  # hardcoded 10x10 grid of images
    ix, iy = 28, 28  # hardcoded for 28x28 MNIST images
    data = tf.reshape(data, [-1])[:gx * gy * ix * iy]
    data = tf.reshape(data, [gy, gx, iy, ix])
    data = tf.transpose(data, [0, 2, 1, 3])
    data = tf.reshape(data, [1, gy * iy, gx * ix, 1])
    return data


####################################################################
# load data

with tf.name_scope("data"):
    data_sets = tf.contrib.learn.datasets.mnist.load_mnist('mnist')
    train_data = data_sets.train
    test_images = binarize(data_sets.test.images)
    visible_size = train_data.images.shape[1]
    total_size = visible_size + HIDDEN_SIZE

    batches_per_epoch = train_data.num_examples // BATCH_SIZE
    assert(batches_per_epoch * BATCH_SIZE == train_data.num_examples)

####################################################################
# variables and placeholders

# model we're training
with tf.name_scope("rbm-variables"):
    rbm_biases = tf.get_variable('rbm_biases', dtype=tf.float32,
                                 shape=[total_size],
                                 initializer=tf.zeros_initializer)
    rbm_weights = tf.get_variable('rbm_weights', dtype=tf.float32,
                                  shape=[visible_size, HIDDEN_SIZE],
                                  initializer=tf.random_normal_initializer(
                                      stddev=0.01))

    # split biases into visible and hidden parts
    rbm_visible_biases = rbm_biases[:visible_size]
    rbm_hidden_biases = rbm_biases[visible_size:]

# visible input
visible_samples = tf.placeholder(dtype=tf.float32,
                                 shape=[None, visible_size],
                                 name='visible_samples')

# global step used by optimizer
global_step = tf.get_variable('global_step', dtype=tf.int32,
                              initializer=0, trainable=False)

####################################################################
# training

#pa = qupa.PopulationAnnealer(visible_size, HIDDEN_SIZE, NUM_SAMPLES)
pa = PCD(left_size=visible_size, right_size=HIDDEN_SIZE,
                   num_samples=NUM_SAMPLES, dtype=tf.float32)

with tf.name_scope('train'):
    train_logz = pa.training_log_z(rbm_biases, rbm_weights,
                                   num_mcmc_sweeps=NUM_MCMC_SWEEPS)
    train_free_energy = mean_free_energy(visible_samples, rbm_visible_biases,
                                         rbm_hidden_biases, rbm_weights)
    train_neg_ll = tf.add(train_logz, train_free_energy, name='train_neg_ll')
    train_regularizer = 2e-4 * tf.nn.l2_loss(rbm_weights)
    train_loss = tf.add(train_neg_ll, train_regularizer, name='train_loss')

    lr = tf.train.exponential_decay(1e-2, global_step,
                                    100 * batches_per_epoch, 0.3, name='lr_decay')
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-2)
        train_op = optimizer.minimize(train_loss, global_step)

    print("trainable variables")
    for op in tf.trainable_variables():
        print(op)

####################################################################
# evaluation

with tf.name_scope('evaluation'):
    data_probs = np.clip(np.mean(train_data.images, axis=0), 0.05, 0.95)
    logz_init_biases = np.concatenate(
        [np.log(data_probs / (1 - data_probs)),
         np.zeros([HIDDEN_SIZE], dtype=np.float32)], axis=0)
    estimate_logz = qupa.evaluation_log_z(
        rbm_biases, rbm_weights, logz_init_biases,
        num_samples=EVAL_NUM_SAMPLES, num_steps=EVAL_NUM_STEPS,
        num_mcmc_sweeps=EVAL_NUM_MCMC_SWEEPS)

    test_free_energy = mean_free_energy(test_images, rbm_visible_biases,
                                        rbm_hidden_biases, rbm_weights)
    test_neg_ll = estimate_logz + test_free_energy

    tf.summary.scalar('estimated log Z', estimate_logz)
    tf.summary.scalar('test mean free energy', test_free_energy)
    tf.summary.scalar('test negative log likelihood', test_neg_ll)

####################################################################
# visualization

with tf.name_scope('images'):
    visible_prob = tf.sigmoid(
        tf.matmul(pa.samples()[:, visible_size:], rbm_weights,
                  transpose_b=True) + rbm_visible_biases)
    tf.summary.image('generated samples', make_image(visible_prob))
    tf.summary.image('weights', make_image(tf.transpose(rbm_weights)))

####################################################################
# run

init = tf.global_variables_initializer()
summaries = tf.summary.merge_all()
tf.get_default_graph().finalize()
fw = tf.summary.FileWriter('logs', graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(NUM_EPOCHS):
        print("Epoch", epoch)
        for _ in range(batches_per_epoch):
            batch = binarize(train_data.next_batch(BATCH_SIZE, shuffle=True)[0])
            sess.run([train_op], feed_dict={visible_samples: batch})

        if epoch % STATS_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
            summary_data, neg_ll_val = sess.run([summaries, test_neg_ll])
            fw.add_summary(summary_data, epoch)
            print("  test neg LL", neg_ll_val)