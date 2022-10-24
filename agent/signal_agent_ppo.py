import threading

import tensorflow as tf
from agent.aiModels.PPO_actor_discrete import APPO_D
from agent.aiModels.PPO_actor_contiuous import APPO_C
from agent.aiModels.PPO_critic import CPPO
from agent.aiModels.state_predictor import SP
import utils.auto_scaling_settings as ACS
from results.metrics import metrics
from utils.actions_definition import ACTIONS
import numpy as np

class PPO:
    def __init__(self):
        self.sess = tf.Session()
        self.obs_dim = ACS.n_node * 8
        self.actor = APPO_C(0, ACS.actors[0][1], self.sess, ACS.actors[0][0], self.obs_dim, ACS.actors[0][2], layers=ACS.actors[0][3])
        self.critic = CPPO(self.sess, self.obs_dim, ACS.critic[0], layers=ACS.critic[1])
        self.a_update_steps = 5
        self.c_update_steps = 5
        self.sess.run(tf.global_variables_initializer())
        self.pending_action, self.pending_state = None, None
        self.buffer_s, self.buffer_r, self.buffer_a = [], [], []
        self.last_obs_id = -1

    def choose_actions(self, obs):
        action = ACTIONS()
        act_value = self.actor.choose_action(obs)
        #action.raw_sch = act_value[0]
        act_value = self.shape_action(act_value)
        action.raw_sch = act_value
        action.sch = act_value.reshape((len(ACS.t_NFs) - 1, ACS.n_node))
        return action

    def shape_action(self, input):
        output = []
        #print(input.shape)
        for i in range(input.shape[0]):
            act_value = input[i,:]
            act_value += 1e-5
            act_value = (np.max(act_value) - act_value) / (np.max(act_value) - np.min(act_value))
            act_value = act_value.reshape((len(ACS.t_NFs) - 1, ACS.n_node))
            for i in range(act_value.shape[0]):
                act_value[i, :] = (act_value[i, :] / np.sum(act_value[i, :]))
            act_value = act_value.flatten()
            output.append(act_value)
        #print(numpy.array(output).shape)
        return np.array(output)

    def update(self, s, a, r):
        action = []
        for i in range(len(a)):
            action.append(a[i].raw_sch)
        ba = np.vstack(action)

        self.sess.run(self.actor.update_oldpi_op)
        adv = self.sess.run(self.critic.advantage, {self.critic.tfs: s, self.critic.tfdc_r: r})
        [self.sess.run(self.actor.atrain_op, {self.actor.tfs: s, self.actor.tfa: ba, self.actor.tfadv: adv}) for _ in range(self.a_update_steps)]
        [self.sess.run(self.critic.ctrain_op, {self.critic.tfs: s, self.critic.tfdc_r: r}) for _ in range(self.c_update_steps)]

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

        if ts == 99: # 200 time steps
            if avai_obs is None:
                v_s_ = self.critic.get_v(self.buffer_s[-1])
            else:
                v_s_ = self.critic.get_v(avai_obs[2])
            #print('s_: value = %f' % (v_s_))
            discounted_r = []
            for r in self.buffer_r[::-1]:
                #print(r)
                v_s_ = r + 0.9 * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()
            # print('discounted_r: ', discounted_r)
            bs, br = np.vstack(self.buffer_s), np.array(discounted_r)[:, np.newaxis]
            self.update(bs, self.buffer_a, br)
            self.buffer_s, self.buffer_a, self.buffer_r = [], [], []
            self.pending_action, self.pending_state = None, None
            self.last_obs_id = -1
            return None

        if avai_obs == None:
            return None # No action
        #print(avai_obs)
        for obs in arrived_obs:
            obs_on_road.remove(obs)
        #print(avai_obs[2])
        action = self.choose_actions(avai_obs[2])

        if self.pending_action is not None:
            self.buffer_s.append(self.pending_state)
            self.buffer_a.append(self.pending_action)
            self.buffer_r.append(avai_obs[3])
            #print(avai_obs)
        self.pending_state = avai_obs[2]
        self.pending_action = action

        return action