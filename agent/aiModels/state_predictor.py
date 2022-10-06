import tensorflow as tf

class SP:
    def __init__(self, sess, obs_dim, act_dim, layers):
        self.sess = sess
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.layers = layers

        self.tfs = tf.placeholder(tf.float32, [None, self.obs_dim], 'pred_inp_state')
        self.tfa = tf.placeholder(tf.float32, [None, self.act_dim], 'pred_ac_action')
        self.tfsa = tf.concat([self.tfs, self.tfa], axis=1, name='pred_concat')

        self.model, self.params = self._build_predict_network()

    def _build_predict_network(self):
        with tf.variable_scope('prediction'):
            for i, layer in enumerate(self.layers):
                if i == 0:
                    l_a = tf.layers.dense(self.tfsa, layer, tf.nn.relu, trainable=True)
                else:
                    l_a = tf.layers.dense(l_a, layer, tf.nn.relu, trainable=True)
            mu = 2 * tf.layers.dense(l_a, self.obs_dim, tf.nn.tanh, trainable=True)
            sigma = tf.layers.dense(l_a, self.obs_dim, tf.nn.softplus, trainable=True)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='prediction')
        return norm_dist, params