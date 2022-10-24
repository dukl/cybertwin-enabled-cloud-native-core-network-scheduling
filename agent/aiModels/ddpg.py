import math
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
import keras
from collections import deque
from utils.logger import log
from utils.actions_definition import ACTIONS
import utils.auto_scaling_settings as ACS

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
        self.epsilon_decay = 0.99985
        self.tau     = 0.001
        self.memory  = deque(maxlen=400000)
        self.update_ite = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.last_obs_id = -1
        self.delayed_reward = None

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.act_dim])
        actor_model_weights    = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(0.0001).apply_gradients(grads)

        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()
        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        self.sess.run(tf.global_variables_initializer())

        self.update_target()

        self.pending_s = None
        self.pending_a = None

    def reset(self):
        self.last_obs_id = -1
        self.pending_s = None
        self.pending_a = None

    def create_actor_model(self):
        state_input = Input(shape=(self.obs_dim,))
        h = Dense(1024, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=math.sqrt(1024), seed=None))(state_input)
        h2 = Dense(1024, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=math.sqrt(1024)))(h)
        output = Dense(self.act_dim, activation='tanh', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(h2)
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.00001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=(self.obs_dim,))
        state_h = Dense(1024, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=math.sqrt(1024)))(state_input)
        state_h2 = Dense(1024, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=math.sqrt(1024)))(state_h)
        action_input = Input(shape=(self.act_dim,))
        action_h = Dense(1024, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=math.sqrt(1024)))(action_input)
        action_h2 = Dense(1024, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=math.sqrt(1024)))(action_h)
        merged = Concatenate()([state_h2, action_h2])
        merged_h1 = Dense(1024, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=math.sqrt(1024)))(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model = Model(inputs=[state_input, action_input], outputs=output)
        adam = Adam(lr=0.0001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def act(self, state):
        act_value = None
        state = state.reshape(1, state.shape[0])
        #log.logger.debug('[line-24][generate action value to be executed]')
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            action = self.actor_model.predict(state)
            #log.logger.debug('%s' % (str(action.tolist())))
            noise = []
            for i in range(action.shape[1]):
                tmp = Ornstein_Uhlenbeck_Noise(sigma=2, mu=np.zeros(1))
                #print('noise = ', tmp())
                noise.append(tmp()[0])
            #log.logger.debug('noise = \n %s' % (str(noise)))
            act_value = action + numpy.array(noise)
            #return action + numpy.array(noise)
        else:
            act_value = self.actor_model.predict(state)
        return act_value#self.shape_action(act_value)

    def shape_action(self, input):
        output = []
        for i in range(input.shape[0]):
            act_value = input[i,:]
            act_value += 1e-5
            act_value = (np.max(act_value) - act_value) / (np.max(act_value) - np.min(act_value))
            act_value = act_value.reshape(len(ACS.t_NFs) - 1, ACS.n_node)
            for i in range(act_value.shape[0]):
                act_value[i, :] = (act_value[i, :] / np.sum(act_value[i, :]))
            act_value = act_value.flatten()
            output.append(act_value)
        #print(numpy.array(output).shape)
        return numpy.array(output)

    def shape_h_s_action(self, input):
        h_s_out = input.reshape((ACS.n_node, len(ACS.t_NFs) - 1))
        h_s_out[h_s_out < -0.6] = -1
        h_s_out[h_s_out > 0.6] = 1
        # print(h_s_out.astype('int32'))
        h_s_out = h_s_out.astype('int32')
        return h_s_out

    def choose_action_with_delayed_obs(self, obs_on_road, ts):
        avai_obs = None
        arrived_obs = []
        for i, obs in enumerate(obs_on_road):
            if obs[0] + obs[1] <= ts:
                arrived_obs.append(obs)
        max_delay = 0
        index, is_avai_obs = 0, False
        for i, obs in enumerate(arrived_obs):
            if obs[0] > self.last_obs_id:
                if obs[0] + obs[1] >= max_delay:
                    max_delay = obs[0] + obs[1]
                    index = i
                    is_avai_obs = True
                    self.last_obs_id = obs[0]
        if is_avai_obs is True:
            avai_obs = arrived_obs[index]
        if avai_obs == None:
            return None # No action

        for obs in arrived_obs:
            obs_on_road.remove(obs)

        #log.logger.debug('receive obs: %s at ts=%d' % (str(avai_obs), ts))
        v_s = self.act(avai_obs[2])
        #print(v_s)
        self.remember(avai_obs[2], v_s, avai_obs[3])

        v_s += 1e-5
        v_s = (np.max(v_s) - v_s) / (np.max(v_s) - np.min(v_s))

        v_s = v_s.reshape((ACS.n_node, (len(ACS.t_NFs) - 1) * ACS.n_max_inst))
        for i in range(v_s.shape[0]):
            v_s[i,:] = v_s[i,:] / np.sum(v_s[i,:])

        ret_a = ACTIONS()

        ret_a.v_s = v_s
        ret_a.h_s = np.zeros((ACS.n_node, len(ACS.t_NFs)-1)).astype('int32')
        #ret_a.h_s = self.shape_h_s_action(v_s)
        #ret_a.sch = ret_a.sch.reshape(len(ACS.t_NFs) - 1, ACS.n_node)
        #print(ret_a.v_s)

        return ret_a

    def remember(self, s, a, r):
        log.logger.debug('remember')
        if self.pending_s is not None:
            self.memory.append([self.pending_s, self.pending_a, r, s])
            #self.pending_s, self.pending_a = s, a
            #print(self.memory[-1])
        self.pending_s, self.pending_a = s, a
        if len(self.memory) >=64 and len(self.memory) % 64 == 0:
            self.train(64)

    def train(self, batch_size):
        log.logger.debug('training ... ')
        samples = random.sample(self.memory, batch_size)
        #print('samples = \n%s', samples)
        self.samples = samples
        self.train_critic(samples)
        self.train_actor(samples)
        self.update_ite += 1
        if self.update_ite % 10 == 0:
            self.update_target()

    def train_critic(self, samples):
        log.logger.debug('[Training critic]')
        s_ts, actions, rewards, s_ts1 = stack_samples(samples)
        #log.logger.debug('s_t = \n%s' % (str(s_ts[0].tolist())))
        #log.logger.debug('s_t+1 = \n%s' % (str(s_ts1[0].tolist())))
        #log.logger.debug('actions = \n%s' % (str(actions.tolist())))
        target_actions = self.target_actor_model.predict(s_ts1)
        #target_actions = self.shape_action(target_actions)
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
        #predicted_actions = self.shape_action(predicted_actions)
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

