import random

import tensorflow as tf
import utils.auto_scaling_settings as ACS
import numpy as np
from utils.actions_definition import ACTIONS

class CRITIC:
    def __init__(self, id, sess, obs_dim, a_h_s, a_v_s, a_sch, lr=0.001, layers=None):
        self.id = id
        self.sess = sess
        self.obs_dim = obs_dim
        self.a_h_s = a_h_s
        self.a_v_s = a_v_s
        self.a_sch = a_sch
        self.lr = lr
        self.layers = layers
        self.tfs = tf.placeholder(tf.float32, [None, self.obs_dim], 'cstate'+str(self.id))
        self.tf_a_h_s = tf.placeholder(tf.float32, [None, self.a_h_s], 'a_h_s'+str(self.id))
        self.tf_a_v_s = tf.placeholder(tf.float32, [None, self.a_v_s], 'a_v_s'+str(self.id))
        self.tf_a_sch = tf.placeholder(tf.float32, [None, self.a_sch], 'a_sch'+str(self.id))

        self.exec_q_a, self.params = self._build_cnet('exec_critic'+str(self.id), True)
        self.targ_q_a, self.targ_params = self._build_cnet('targ_critic'+str(self.id), False)

        self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.exec_q_a))
        self.critic_train = tf.train.AdamOptimizer(self.lr).minimize(self.critic_loss)

        self.grad_a = [tf.gradients(self.exec_q_a,self.tf_a_h_s),tf.gradients(self.exec_q_a, self.tf_a_v_s),tf.gradients(self.exec_q_a, self.tf_a_sch)]
        #self.grad_a_h_s = tf.gradients(self.exec_q_a, self.tf_a_h_s)
        #self.grad_a_v_s = tf.gradients(self.exec_q_a, self.tf_a_v_s)
        #self.grad_a_sch = tf.gradients(self.exec_q_a, self.tf_a_sch)

        self.tau = 0.01

        with tf.variable_scope('update_critic_target'+str(self.id)):
            self.update_critic_target = [oldp.assign(p*self.tau+oldp*(1-self.tau)) for p ,oldp in zip(self.params, self.targ_params)]

    def _build_cnet(self, name, trainable):
        with tf.variable_scope(name):
            lc = tf.concat([self.tfs, self.tf_a_h_s, self.tf_a_v_s, self.tf_a_sch], axis=1)
            for i, layer in enumerate(self.layers):
                lc = tf.layers.dense(lc, layer, tf.nn.relu, trainable=trainable)
            v = tf.layers.dense(lc, 1, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return v, params

class ACTOR:
    def __init__(self, id, sess, act_dim, obs_dim, lr=0.001, layers=None):
        self.id = id
        self.sess = sess
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.lr = lr
        self.layers = layers
        self.tfs = tf.placeholder(tf.float32, [None, self.obs_dim], 'ac_state_'+str(self.id))
        self.exec_actor, self.params = self._build_anet('exec-actor-'+str(self.id) ,trainable=True)
        self.targ_actor, self.targ_params = self._build_anet('targ-actor-' + str(self.id), trainable=False)

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.act_dim])
        self.actor_grads = tf.gradients(self.exec_actor, self.params, -self.actor_critic_grad)
        grads = zip(self.actor_grads, self.params)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)

        self.tau = 0.01

        with tf.variable_scope('update_actor_target_'+str(self.id)):
            self.update_actor_target = [oldp.assign(p*self.tau + oldp*(1-self.tau)) for p ,oldp in zip(self.params, self.targ_params)]

        self.epsilon = 0.9
        self.epsilon_decay = 0.99995


    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            for i, layer in enumerate(self.layers):
                if i == 0:
                    l_a = tf.layers.dense(self.tfs, layer, tf.nn.relu, trainable=trainable)
                else:
                    l_a = tf.layers.dense(l_a, layer, tf.nn.relu, trainable=trainable)
            output = tf.layers.dense(l_a, self.act_dim, tf.nn.tanh, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return output, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.sess.run(self.exec_actor, {self.tfs: s})[0] * 2 + np.random.normal()
        return self.sess.run(self.exec_actor, {self.tfs: s})[0]


class MADDPG1:
    def __init__(self):
        self.sess = tf.Session()
        self.obs_dim = ACS.n_node * 8
        self.actor_h_s = None
        self.critic_h_s = None
        self.actor_v_s = None
        self.critic_v_s = None
        self.actor_sch = None
        self.critic_sch = None
        self.actors = []
        self.critic = []
        self._build_actor_critic_network()

        self.gamma = 0.99

        self.sess.run(tf.global_variables_initializer())

        self.memory = []
        self.update_it = 0

    def _build_actor_critic_network(self):
        act_dim = []
        for i, params in enumerate(ACS.actors):
            act_dim.append(params[0])
            if i == 0:
                self.actor_h_s = ACTOR(i, self.sess, params[0], self.obs_dim, params[2], layers=params[3])
                self.actors.append(self.actor_h_s)
                #self.critic_h_s = CRITIC(i, self.sess, self.obs_dim, act_dim[0], act_dim[1], act_dim[2], ACS.critic[0], layers=ACS.critic[1])
                #self.critic.append(self.critic_h_s)
            if i == 1:
                self.actor_v_s = ACTOR(i, self.sess, params[0], self.obs_dim, params[2], layers=params[3])
                self.actors.append(self.actor_v_s)
                #self.critic_v_s = CRITIC(i, self.sess, self.obs_dim, act_dim[0], act_dim[1], act_dim[2], ACS.critic[0], layers=ACS.critic[1])
                #self.critic.append(self.critic_v_s)
            if i == 2:
                self.actor_sch = ACTOR(i, self.sess, params[0], self.obs_dim, params[2], layers=params[3])
                self.actors.append(self.actor_sch)
                #self.critic_sch = CRITIC(i, self.sess, self.obs_dim, act_dim[0], act_dim[1], act_dim[2], ACS.critic[0], layers=ACS.critic[1])
                #self.critic.append(self.critic_sch)
        for i in range(len(ACS.actors)):
            critic = CRITIC(i, self.sess, self.obs_dim, act_dim[0], act_dim[1], act_dim[2], ACS.critic[0], layers=ACS.critic[1])
            self.critic.append(critic)
        #self.critic = CRITIC(self.sess, self.obs_dim, act_dim[0], act_dim[1], act_dim[2], ACS.critic[0], layers=ACS.critic[1])

    def choose_actions(self, obs):
        action = ACTIONS()
        h_s_out = self.actor_h_s.choose_action(obs)
        #print(h_s_out.shape)
        h_s_out = h_s_out.reshape((ACS.n_node, len(ACS.t_NFs) - 1))
        h_s_out[h_s_out < -0.6] = -1
        h_s_out[h_s_out > 0.6] = 1
        #print(h_s_out.astype('int32'))
        action.h_s = h_s_out.astype('int32')
        action.raw_h_s = h_s_out.reshape((ACS.n_node*(len(ACS.t_NFs)-1))).astype('float32')
        #print(action.raw_h_s.shape)
        v_s_out = self.actor_v_s.choose_action(obs)
        v_s_out = v_s_out.reshape((ACS.n_node * (len(ACS.t_NFs) - 1), ACS.n_max_inst))
        v_s_out = np.clip(v_s_out, -1, 1)
        #print(v_s_out)
        action.v_s = v_s_out
        action.raw_v_s = v_s_out.reshape((ACS.n_node*(len(ACS.t_NFs)-1)*ACS.n_max_inst)).astype('float32')
        scheduling = self.actor_sch.choose_action(obs) + 1e-5
        scheduling = (np.max(scheduling) - scheduling) / (np.max(scheduling) - np.min(scheduling))
        scheduling = scheduling.reshape((len(ACS.msg_msc) * (len(ACS.t_NFs) - 1), ACS.n_node))
        #print(scheduling)
        action.sch = scheduling
        action.raw_sch = scheduling.reshape((len(ACS.msg_msc)*(len(ACS.t_NFs)-1)*ACS.n_node)).astype('float32')
        return action

    def train_critic(self, samples):
        s, a_h_s, a_v_s, a_sch, r, s_ = [], [], [], [], [], []
        #samples = np.array(random.sample(self.memory, batch_size))
        for sample in samples:
            s.append(sample[0])
            a_h_s.append(sample[1])
            a_v_s.append(sample[2])
            a_sch.append(sample[3])
            r.append(sample[4])
            s_.append(sample[5])
        #print(s_)
        s, a_h_s, a_v_s, a_sch, r, s_ = np.vstack(s), np.vstack(a_h_s), np.vstack(a_v_s), np.vstack(a_sch), np.vstack(r), np.vstack(s_)
        #print(s_.shape)
        targ_a_h_s = self.sess.run(self.actor_h_s.targ_actor, feed_dict={self.actor_h_s.tfs:s})
        targ_a_v_s = self.sess.run(self.actor_v_s.targ_actor, feed_dict={self.actor_v_s.tfs:s})
        targ_a_sch = self.sess.run(self.actor_sch.targ_actor, feed_dict={self.actor_sch.tfs:s})
        #critic_input = np.concatenate([s, targ_a_h_s, targ_a_v_s, targ_a_sch], axis=0)
        future_rewards = self.sess.run(self.critic.targ_q_a, feed_dict={self.critic.tfs:s_, self.critic.tf_a_h_s:targ_a_h_s, self.critic.tf_a_v_s:targ_a_v_s, self.critic.tf_a_sch:targ_a_sch})
        r += self.gamma * future_rewards
        self.sess.run(self.critic.critic_train, feed_dict={self.critic.tfs:s, self.critic.tf_a_h_s:a_h_s, self.critic.tf_a_v_s:a_v_s, self.critic.tf_a_sch:a_sch, self.critic.target_Q:r})



    def train_actor(self, samples):
        s, a_h_s, a_v_s, a_sch, r, s_ = [], [], [], [], [], []
        # samples = np.array(random.sample(self.memory, batch_size))
        for sample in samples:
            s.append(sample[0])
            a_h_s.append(sample[1])
            a_v_s.append(sample[2])
            a_sch.append(sample[3])
            r.append(sample[4])
            s_.append(sample[5])
        s, a_h_s, a_v_s, a_sch, r, s_ = np.vstack(s), np.vstack(a_h_s), np.vstack(a_v_s), np.vstack(a_sch), np.vstack(r), np.vstack(s_)
        p_a_h_s = self.sess.run(self.actor_h_s.exec_actor, feed_dict={self.actor_h_s.tfs:s})
        p_a_v_s = self.sess.run(self.actor_v_s.exec_actor, feed_dict={self.actor_v_s.tfs:s})
        p_a_sch = self.sess.run(self.actor_sch.exec_actor, feed_dict={self.actor_sch.tfs:s})
        grad_a_h_s = self.sess.run(self.critic.grad_a_h_s, feed_dict={
            self.critic.tfs:s,
            self.critic.tf_a_h_s:p_a_h_s,
            self.critic.tf_a_v_s:p_a_v_s,
            self.critic.tf_a_sch:p_a_sch
        })[0]
        grad_a_v_s = self.sess.run(self.critic.grad_a_v_s, feed_dict={
            self.critic.tfs: s,
            self.critic.tf_a_h_s: p_a_h_s,
            self.critic.tf_a_v_s: p_a_v_s,
            self.critic.tf_a_sch: p_a_sch
        })[0]
        grad_a_sch = self.sess.run(self.critic.grad_a_sch, feed_dict={
            self.critic.tfs: s,
            self.critic.tf_a_h_s: p_a_h_s,
            self.critic.tf_a_v_s: p_a_v_s,
            self.critic.tf_a_sch: p_a_sch
        })[0]
        self.sess.run(self.actor_h_s.optimize, feed_dict={
            self.actor_h_s.tfs:s,
            self.actor_h_s.actor_critic_grad:grad_a_h_s
        })
        self.sess.run(self.actor_v_s.optimize, feed_dict={
            self.actor_v_s.tfs: s,
            self.actor_v_s.actor_critic_grad: grad_a_v_s
        })
        self.sess.run(self.actor_sch.optimize, feed_dict={
            self.actor_sch.tfs: s,
            self.actor_sch.actor_critic_grad: grad_a_sch
        })

    def train(self, batch_size):
        for i in range (len(ACS.actors)):
            s, a_h_s, a_v_s, a_sch, r, s_ = [], [], [], [], [], []
            samples = np.array(random.sample(self.memory, batch_size))
            for sample in samples:
                s.append(sample[0])
                a_h_s.append(sample[1])
                a_v_s.append(sample[2])
                a_sch.append(sample[3])
                r.append(sample[4])
                s_.append(sample[5])
            s, a_h_s, a_v_s, a_sch, r, s_ = np.vstack(s), np.vstack(a_h_s), np.vstack(a_v_s), np.vstack(a_sch), np.vstack(r), np.vstack(s_)
            targ_a_h_s = self.sess.run(self.actor_h_s.targ_actor, feed_dict={self.actor_h_s.tfs: s})
            targ_a_v_s = self.sess.run(self.actor_v_s.targ_actor, feed_dict={self.actor_v_s.tfs: s})
            targ_a_sch = self.sess.run(self.actor_sch.targ_actor, feed_dict={self.actor_sch.tfs: s})
            # critic_input = np.concatenate([s, targ_a_h_s, targ_a_v_s, targ_a_sch], axis=0)
            future_rewards = self.sess.run(self.critic[i].targ_q_a,
                                           feed_dict={self.critic[i].tfs: s_, self.critic[i].tf_a_h_s: targ_a_h_s,
                                                      self.critic[i].tf_a_v_s: targ_a_v_s,
                                                      self.critic[i].tf_a_sch: targ_a_sch})
            r += self.gamma * future_rewards
            self.sess.run(self.critic[i].critic_train,
                          feed_dict={self.critic[i].tfs: s, self.critic[i].tf_a_h_s: a_h_s, self.critic[i].tf_a_v_s: a_v_s,
                                     self.critic[i].tf_a_sch: a_sch, self.critic[i].target_Q: r})
            p_a_h_s = self.sess.run(self.actor_h_s.exec_actor, feed_dict={self.actor_h_s.tfs: s})
            p_a_v_s = self.sess.run(self.actor_v_s.exec_actor, feed_dict={self.actor_v_s.tfs: s})
            p_a_sch = self.sess.run(self.actor_sch.exec_actor, feed_dict={self.actor_sch.tfs: s})
            grad = self.sess.run(self.critic[i].grad_a[i], feed_dict={
                self.critic[i].tfs: s,
                self.critic[i].tf_a_h_s: p_a_h_s,
                self.critic[i].tf_a_v_s: p_a_v_s,
                self.critic[i].tf_a_sch: p_a_sch
            })[0]
            #print(grad.shape)
            #print('i=%d'%(i))
            self.sess.run(self.actors[i].optimize, feed_dict={
                self.actors[i].tfs: s,
                self.actors[i].actor_critic_grad: grad
            })

    def no_delay_memory(self, s, a, r, s_):
        batch_size = 32
        a_h_s = a.raw_h_s
        a_v_s = a.raw_v_s
        a_sch = a.raw_sch
        self.memory.append([s,a_h_s,a_v_s,a_sch,r,s_])
        if len(self.memory) > 100 and len(self.memory) % batch_size == 0:
            self.train(batch_size)
            #print('training ...')
            #s, a_h_s, a_v_s, a_sch, r, s_ = [], [], [], [], [], []
            #samples = np.array(random.sample(self.memory, batch_size))
            #self.train_critic(samples)
            #self.train_actor(samples)
            self.update_it += 1
            if self.update_it >= 10 and self.update_it % 10 == 0:
                #print('updating ...')
                for i in range(len(ACS.actors)):
                    self.sess.run(self.actors[i].update_actor_target)
                    #self.sess.run(self.actor_v_s.update_actor_target)
                    #self.sess.run(self.actor_sch.update_actor_target)
                    self.sess.run(self.critic[i].update_critic_target)
