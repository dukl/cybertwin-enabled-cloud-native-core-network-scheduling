import random

import numpy
import numpy as np

import utils.global_parameters as GP
from utils.obs_reward_action_def import ACT
import os,sys
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
import keras.backend as K
from collections import deque
from utils.logger import log


class Ornstein_Uhlenbeck_Noise:
    def __init__(self, mu, sigma=1.0, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        '''
        后两行是dXt，其中后两行的前一行是θ(μ-Xt)dt，后一行是σεsqrt(dt)
        '''
        self.x_prev = x
        return x

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu)

def stack_samples(samples):
    array = np.array(samples)
    s_ts = np.stack(array[:,0]).reshape((array.shape[0], -1))
    actions = np.stack(array[:, 1]).reshape((array.shape[0], -1))
    rewards = np.stack(array[:, 2]).reshape((array.shape[0], -1))
    s_ts1 = np.stack(array[:, 3]).reshape((array.shape[0], -1))
    return s_ts, actions, rewards, s_ts1

class DDPG:
    def __init__(self, obs_dim, act_dim):
        self.sess = tf.Session()
        self.epsilon = 0.9
        self.gamma   = 0.99
        self.epsilon_decay = 0.9995
        self.tau     = 0.001
        self.memory  = deque(maxlen=400000)
        self.update_ite = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.act_dim])
        actor_model_weights    = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(0.001).apply_gradients(grads)

        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()
        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        self.sess.run(tf.global_variables_initializer())

        self.update_target()

        self.pending_s = None
        self.pending_a = None

    def create_actor_model(self):
        state_input = Input(shape=(self.obs_dim,))
        h = Dense(1024, activation='relu')(state_input)
        h = Dense(1024, activation='relu')(h)
        h2 = Dense(1024, activation='relu')(h)
        output = Dense(self.act_dim, activation='tanh')(h2)
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=(self.obs_dim,))
        state_h = Dense(1024, activation='relu')(state_input)
        state_h = Dense(1024)(state_h)
        state_h2 = Dense(1024)(state_h)
        action_input = Input(shape=(self.act_dim,))
        action_h = Dense(1024)(action_input)
        action_h = Dense(1024)(action_h)
        action_h2 = Dense(1024)(action_h)
        merged = Concatenate()([state_h2, action_h2])
        merged_h1 = Dense(1024, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model = Model(inputs=[state_input, action_input], outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def act(self, state):
        state = state.reshape(1, state.shape[0])
        #log.logger.debug('[line-24][generate action value to be executed]')
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            action = self.actor_model.predict(state)
            #log.logger.debug('%s' % (str(action.tolist())))
            noise = []
            for i in range(action.shape[1]):
                tmp = Ornstein_Uhlenbeck_Noise(sigma=5, mu=np.zeros(1))
                noise.append(tmp()[0])
            #log.logger.debug('noise = \n %s' % (str(noise)))
            return action + numpy.array(noise)
        return self.actor_model.predict(state)

    def remember(self, s, a, r):
        if self.pending_s is not None:
            self.memory.append([self.pending_s, self.pending_a, r, s])
            self.pending_s, self.pending_a = s, a
        else:
            self.pending_s, self.pending_a = s, a

    def train(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        self.samples = samples
        self.train_critic(samples)
        self.train_actor(samples)

    def train_critic(self, samples):
        log.logger.debug('[Training critic]')
        s_ts, actions, rewards, s_ts1 = stack_samples(samples)
        #log.logger.debug('s_t = \n%s' % (str(s_ts[0].tolist())))
        #log.logger.debug('s_t+1 = \n%s' % (str(s_ts1[0].tolist())))
        #log.logger.debug('actions = \n%s' % (str(actions.tolist())))
        target_actions = self.target_actor_model.predict(s_ts1)
        target_actions = (target_actions - numpy.min(target_actions)) / (numpy.max(target_actions) - numpy.min(target_actions)) * GP.n_servers * GP.n_ms_server * GP.ypi_max
        target_actions = target_actions.astype('int')
        #log.logger.debug('target_actions = \n %s' % (str(target_actions.tolist())))
        future_rewards = self.target_critic_model.predict([s_ts1, target_actions])
        #log.logger.debug('train_critic, future_rewards = \n%s' % (str(future_rewards.tolist())))
        #log.logger.debug('reward = \n%s' % (str(rewards.tolist())))
        rewards += self.gamma*future_rewards
        train_history = self.critic_model.fit([s_ts, actions], rewards, verbose=0)
        q_loss = train_history.history['loss'][0]
        log.logger.debug('Q_loss = %s' % (str(q_loss)))

    def train_actor(self, samples):
        log.logger.debug('[Training actor]')
        s_ts, _, _, _ = stack_samples(samples)
        predicted_actions = self.actor_model.predict(s_ts)
        predicted_actions = (predicted_actions - numpy.min(predicted_actions)) / (numpy.max(predicted_actions) - numpy.min(predicted_actions)) * GP.n_servers * GP.n_ms_server * GP.ypi_max
        predicted_actions = predicted_actions.astype('int')
        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_state_input: s_ts,
            self.critic_action_input: predicted_actions
        })[0]
        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: s_ts,
            self.actor_critic_grad: grads
        })

    def update_target(self):
        self.update_actor_target()
        self.update_critic_target()

    def update_actor_target(self):
        log.logger.debug('[Update actor target]')
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]* self.tau + actor_target_weights[i]*(1-self.tau)
        self.target_actor_model.set_weights(actor_target_weights)

    def update_critic_target(self):
        log.logger.debug('[Update critic target]')
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
        self.target_critic_model.set_weights(critic_target_weights)

