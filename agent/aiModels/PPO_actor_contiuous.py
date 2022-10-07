#import tensorflow.compat.v1 as tf
#import tensorflow_probability as tfp
#tf.compat.v1.disable_eager_execution()
import tensorflow as tf

import numpy as np

class APPO_C:
    def __init__(self, id, flag, sess, act_dim, obs_dim, lr=0.001, epsilon=0.2, layers=None):
        self.id = id
        self.sess = sess
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.lr = lr
        self.epsilon = epsilon
        self.layers = layers
        self.is_dicrete = flag

        self.tfs = tf.placeholder(tf.float32, [None, self.obs_dim], 'ac_state_'+str(self.id))
        self.pi, self.pi_params = self._build_anet('ac_pi_'+str(self.id), trainable=True)
        self.oldpi, self.oldpi_params = self._build_anet('ac_oldpi_'+str(self.id), trainable=False)
        with tf.variable_scope('sample_action_'+str(self.id)):
            self.sample_op = tf.squeeze(self.pi.sample(1), axis=0)
        with tf.variable_scope('update_oldpi_'+str(self.id)):
            self.update_oldpi_op = [oldp.assign(p) for p ,oldp in zip(self.pi_params, self.oldpi_params)]
        self.tfa = tf.placeholder(tf.float32, [None, self.act_dim], 'ac_action_'+str(self.id))
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'ac_advantage_'+str(self.id))
        with tf.variable_scope('loss_'+str(self.id)):
            with tf.variable_scope('surrogate_'+str(self.id)):
                self.ratio = self.pi.prob(self.tfa) / (self.oldpi.prob(self.tfa) + 1e-5)
                surr = self.ratio * self.tfadv
            self.aloss = -tf.reduce_mean(tf.minimum(
                surr,
                tf.clip_by_value(self.ratio, 1. - self.epsilon, 1. + self.epsilon) * self.tfadv
            ))
        with tf.variable_scope('atrain_'+str(self.id)):
            self.atrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.aloss)


    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            for i, layer in enumerate(self.layers):
                if i == 0:
                    l_a = tf.layers.dense(self.tfs, layer, tf.nn.relu, trainable=trainable)
                else:
                    l_a = tf.layers.dense(l_a, layer, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l_a, self.act_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l_a, self.act_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        #print('action_out_con_before ', a.tolist())
        #a = (np.max(a) - a) / (np.max(a) - np.min(a))
        #print('action_out_con ', a.tolist())
        return a

