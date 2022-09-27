#import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_eager_execution()
import tensorflow as tf
import numpy as np


class CPPO:
    def __init__(self, sess, obs_dim, lr=0.001, layers=None):
        self.sess = sess
        self.obs_dim = obs_dim
        self.lr = lr
        self.lyers = layers

        self.tfs = tf.placeholder(tf.float32, [None, self.obs_dim], 'cstate')
        w_init = tf.random_normal_initializer(0., .1)
        for i, layer in enumerate(self.lyers):
            if i == 0:
                lc = tf.layers.dense(self.tfs, layer, tf.nn.relu, kernel_initializer=w_init, name='lc'+str(i))
            else:
                lc = tf.layers.dense(lc, layer, tf.nn.relu, kernel_initializer=w_init, name='lc'+str(i))
        self.v = tf.layers.dense(lc, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'dicounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.closs)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]