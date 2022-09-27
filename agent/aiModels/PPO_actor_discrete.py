#import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_eager_execution()
import tensorflow as tf
import numpy as np

class APPO_D:
    def __init__(self, id, sess, act_dim, obs_dim, lr=0.001, epsilon=0.2, layers=None):
        self.id = id
        self.sess = sess
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.lr = lr
        self.epsilon = epsilon
        self.layers = layers

        self.tfs = tf.placeholder(tf.float32, [None, self.obs_dim], 'ad_state_'+str(self.id))
        self.pi, self.pi_params = self._build_anet('ad_pi_'+str(self.id), trainable=True)
        self.oldpi, self.oldpi_params = self._build_anet('ad_oldpi_'+str(self.id), trainable=False)
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_params, self.oldpi_params)]
        self.tfa = tf.placeholder(tf.int32, [None, ],  'ad_action_'+str(self.id))
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'ad_advantage_'+str(self.id))
        self.a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
        pi_prob = tf.gather_nd(params=self.pi, indices=self.a_indices)
        oldpi_prob = tf.gather_nd(params=self.oldpi, indices=self.a_indices)
        ratio = pi_prob/(oldpi_prob + 1e-5)
        surr = ratio * self.tfadv

        self.aloss = -tf.reduce_mean(tf.minimum(
            surr,
            tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * self.tfadv
        ))
        self.atrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.aloss)

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            for i, layer in enumerate(self.layers):
                if i == 0:
                    l_a = tf.layers.dense(self.tfs, layer, tf.nn.relu, trainable=trainable)
                else:
                    l_a = tf.layers.dense(l_a, layer, tf.nn.relu, trainable=trainable)
            a_prob = tf.layers.dense(l_a, self.act_dim, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return a_prob, params

    def choose_action(self, s):
        prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s[None, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())
        return action



