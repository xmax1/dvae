import tensorflow as tf
import numpy as np
import os

import qupa
from qupa.pcd import PCD


num_left_vars = 10
num_right_vars = 5

num_samples = 100

sampler = PCD(left_size=num_left_vars, right_size=num_right_vars, num_samples=num_samples,
              dtype=tf.float32)

with tf.name_scope("rbm-variables"):
    # bias on the left side
    b1 = tf.Variable(tf.zeros(shape=[num_left_vars, 1], dtype=tf.float32), name='bias1')
    # bias on the right side
    b2 = tf.Variable(tf.zeros(shape=[num_right_vars, 1], dtype=tf.float32), name='bias2')
    # pairwise weight
    w = tf.Variable(tf.zeros(shape=[num_left_vars, num_right_vars], dtype=tf.float32), name='pairwise')

    b = tf.concat(values=[tf.squeeze(b1), tf.squeeze(b2)], axis=0, name='biases')

log_z_train = sampler.training_log_z(b, w)

with tf.name_scope("samples"):
    with tf.control_dependencies([log_z_train]):
        samples = sampler.samples()


dir = './logs-test/'
if not os.path.exists(dir):
    os.makedirs(dir)
tfwriter = tf.summary.FileWriter(dir)
summary_op = tf.summary.merge_all()

sess = tf.InteractiveSession()

#print(sess.run(samples))
tfwriter.add_graph(sess.graph)

