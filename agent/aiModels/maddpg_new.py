import tensorflow as tf
import tensorflow.contrib as tc
import utils.auto_scaling_settings as ACS

class MADDPG():
    def __init__(self, name, layer_norm=False, nb_actions=2, nb_input=16, nb_other_aciton=4, actor_id=None):
        gamma = 0.999
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        state_input = tf.placeholder(shape=[None, nb_input], dtype=tf.float32)
        action_input = tf.placeholder(shape=[None, nb_actions], dtype=tf.float32)
        other_action_input = tf.placeholder(shape=[None, nb_other_aciton], dtype=tf.float32)
        reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        #self.layers = ACS.actors[actor_id][3]
        self.critic_layers = ACS.critic[1]
        self.actor_id = actor_id


        def actor_network(name):
            with tf.variable_scope(name) as scope:
                x = state_input
                for i, layer in enumerate(ACS.actors[self.actor_id][3]):
                    x = tf.layers.dense(x, layer)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    x = tf.nn.relu(x)
                x = tf.layers.dense(x, self.nb_actions) #kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                x = tf.nn.tanh(x)
            return x


        def critic_network(name, action_input, reuse=False):
            with tf.variable_scope(name) as scope:
                if reuse:
                    scope.reuse_variables()

                x = state_input
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.concat([x, action_input], axis=-1)
                for i, layer in enumerate(self.critic_layers):
                    x = tf.layers.dense(x, layer)
                    if self.layer_norm:
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    x = tf.nn.relu(x)

                x = tf.layers.dense(x, 1)#, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
            return x


        self.action_output = actor_network(name + "actor")
        self.critic_output = critic_network(name + '_critic',
                                            action_input=tf.concat([action_input, other_action_input], axis=1))
        self.state_input = state_input
        self.action_input = action_input
        self.other_action_input = other_action_input
        self.reward = reward

        self.actor_optimizer = tf.train.AdamOptimizer(ACS.actors[self.actor_id][2])
        self.critic_optimizer = tf.train.AdamOptimizer(ACS.critic[0])

        # ?????????Q???
        self.actor_loss = -tf.reduce_mean(
            critic_network(name + '_critic', action_input=tf.concat([self.action_output, other_action_input], axis=1),
                           reuse=True))
        self.actor_train = self.actor_optimizer.minimize(self.actor_loss)

        self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.critic_output))
        self.critic_train = self.critic_optimizer.minimize(self.critic_loss)

    def train_actor(self, state, other_action, sess):
        sess.run(self.actor_train, {self.state_input: state, self.other_action_input: other_action})

    def train_critic(self, state, action, other_action, target, sess):
        sess.run(self.critic_train,
                 {self.state_input: state, self.action_input: action, self.other_action_input: other_action,
                  self.target_Q: target})

    def action(self, state, sess):
        return sess.run(self.action_output, {self.state_input: state})

    def Q(self, state, action, other_action, sess):
        return sess.run(self.critic_output,
                        {self.state_input: state, self.action_input: action, self.other_action_input: other_action})


