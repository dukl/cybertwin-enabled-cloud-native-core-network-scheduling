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


class HADC:
    def __init__(self):
        self.sess = tf.Session()
        self.obs_dim = ACS.n_node * 8
        self.actor_h_s = None
        self.actor_v_s = None
        self.actor_sch = None
        self.actors = []
        self.critic = None
        #self._build_actor_critic_networks()
        self._build_networks()
        self.a_update_steps = 5
        self.c_update_steps = 5

        self.sess.run(tf.global_variables_initializer())

        self.pending_action, self.pending_state = None, None
        self.buffer_s, self.buffer_r, self.buffer_a = [], [], []

    def _build_networks(self):
        for i, params in enumerate(ACS.actors):
            if i == 0:
                self.actor_h_s = APPO_C(i, params[1], self.sess, params[0], self.obs_dim, params[2], layers=params[3])
                self.actors.append(self.actor_h_s)
            elif i == 1:
                self.actor_v_s = APPO_C(i, params[1], self.sess, params[0], self.obs_dim, params[2], layers=params[3])
                self.actors.append(self.actor_v_s)
            elif i == 2:
                self.actor_sch = APPO_C(i, params[1], self.sess, params[0], self.obs_dim, params[2], layers=params[3])
                self.actors.append(self.actor_sch)
        self.critic = CPPO(self.sess, self.obs_dim, ACS.critic[0], layers=ACS.critic[1])

    def _build_actor_critic_networks(self):
        print('building 1')
        for i in range(ACS.n_node):
            for j in range(len(ACS.t_NFs) - 1):
                for k in range(len(ACS.actors) - 1):
                    params = ACS.actors[k]
                    if params[1] == True:
                        print('True', i, j)
                        actor = APPO_D(i*(len(ACS.t_NFs)-1) + j, params[1], self.sess, params[0], self.obs_dim, params[2], layers=params[3])
                        self.actor_h_s.append(actor)
                    else:
                        print('False', i, j)
                        actor = APPO_C(i*(len(ACS.t_NFs)-1) + j, params[1], self.sess, params[0], self.obs_dim, params[2], layers=params[3])
                        self.actor_v_s.append(actor)
        params = ACS.actors[2]
        print('building 2')
        for i in range(len(ACS.msg_msc)):
            for j in range(len(ACS.t_NFs) - 1):
                actor = APPO_C(i*(len(ACS.t_NFs) - 1) + j + ACS.n_node*(len(ACS.t_NFs)-1), params[1], self.sess, params[0], self.obs_dim, params[2], layers=params[3])
                self.actor_sch.append(actor)
        print('building 3')
        self.critic = CPPO(self.sess, self.obs_dim, ACS.critic[0], layers=ACS.critic[1])

    def actor_v_scaling(self, actor, v_s, i, j, obs):
        v_s[i*(len(ACS.t_NFs) - 1)+j] = np.clip(actor.choose_action(obs), -1, 1)
    def actor_h_scaling(self, actor, h_s, i, j, obs):
        h_s[i,j] = actor.choose_action(obs)
    def actor_scheduling(self, actor, sche, i, j, obs):
        tmp = actor.choose_action(obs) + 1e-3
        sche[i * (len(ACS.t_NFs) - 1) + j] = tmp / np.sum(tmp)

    def choose_actions_old(self, obs):
        action = ACTIONS()
        act_value = []
        h_scaling = np.zeros((ACS.n_node, len(ACS.t_NFs) - 1))
        v_scaling = [np.zeros(ACS.n_max_inst) for _ in range(ACS.n_node*(len(ACS.t_NFs) - 1))]
        scheduling = [np.zeros(ACS.n_node) for _ in range(len(ACS.msg_msc)*(len(ACS.t_NFs) - 1))]

        thread_list = []

        for i in range(ACS.n_node):
            for j in range(len(ACS.t_NFs) - 1):
                h_scaling[i,j] = self.actor_h_s[i*(len(ACS.t_NFs)-1) + j].choose_action(obs)
                v_scaling[i*(len(ACS.t_NFs)-1)+j] = np.clip(self.actor_v_s[i*(len(ACS.t_NFs)-1) + j].choose_action(obs), -1, 1)
                #t_ = threading.Thread(target=self.actor_h_scaling(self.actor_h_s[i*(len(ACS.t_NFs)-1) + j], h_scaling, i, j, obs))
                #thread_list.append(t_)
                #t_.start()
                #t_ = threading.Thread(target=self.actor_v_scaling(self.actor_v_s[i*(len(ACS.t_NFs)-1) + j], v_scaling, i, j, obs))
                #thread_list.append(t_)
                #t_.start()

        for i in range(len(ACS.msg_msc)):
            for j in range(len(ACS.t_NFs) - 1):
                tmp = self.actor_sch[i*(len(ACS.t_NFs)-1) + j].choose_action(obs) + 1e-3
                scheduling[i*(len(ACS.t_NFs)-1)+j] = (tmp/np.sum(tmp))
                #t_ = threading.Thread(target=self.actor_scheduling(self.actor_sch[i*(len(ACS.t_NFs)-1) + j], scheduling, i, j, obs))
                #thread_list.append(t_)
                #t_.start()

        #for j in thread_list:
            #j.join()

        print(h_scaling)
        print(v_scaling)
        print(scheduling)

        action.h_s = h_scaling
        action.v_s = v_scaling
        action.sch = scheduling
        return action

    def choose_actions(self, obs):
        action = ACTIONS()
        h_s_out = self.actor_h_s.choose_action(obs)
        action.raw_h_s = h_s_out
        h_s_out = h_s_out.reshape((ACS.n_node, len(ACS.t_NFs)-1))
        h_s_out[h_s_out<-0.6] = -1
        h_s_out[h_s_out>0.6] = 1
        #print(h_s_out.astype('int32'))
        action.h_s = h_s_out.astype('int32')
        v_s_out = self.actor_v_s.choose_action(obs)
        action.raw_v_s = v_s_out
        v_s_out = v_s_out.reshape((ACS.n_node*(len(ACS.t_NFs)-1), ACS.n_max_inst))
        v_s_out = np.clip(v_s_out, -1, 1)
        #print(v_s_out.shape)
        action.v_s = v_s_out
        scheduling = self.actor_sch.choose_action(obs) + 1e-5
        action.raw_sch = scheduling
        scheduling = (np.max(scheduling) - scheduling)/(np.max(scheduling) - np.min(scheduling))
        scheduling = scheduling.reshape((len(ACS.msg_msc)*(len(ACS.t_NFs)-1), ACS.n_node))
        #print(scheduling.shape)
        action.sch = scheduling
        return action

    def update(self, s, a, r):
        action = [[] for _ in range(len(ACS.actors))]
        for i in range(len(a)):
            action[0].append(a[i].raw_h_s)
            action[1].append(a[i].raw_v_s)
            action[2].append(a[i].raw_sch)

        ba = [[] for _ in range(len(ACS.actors))]
        for i in range(len(ACS.actors)):
            ba[i] = np.vstack(action[i])
        for i, actor in enumerate(self.actors):
            self.sess.run(actor.update_oldpi_op)
            adv = self.sess.run(self.critic.advantage, {self.critic.tfs: s, self.critic.tfdc_r: r})
            [self.sess.run(actor.atrain_op, {actor.tfs: s, actor.tfa: ba[i], actor.tfadv: adv}) for _ in range(self.a_update_steps)]
        [self.sess.run(self.critic.ctrain_op, {self.critic.tfs: s, self.critic.tfdc_r: r}) for _ in range(self.c_update_steps)]


    def choose_action_with_delayed_obs(self, obs_on_road, ts):
        avai_obs = None
        for i, obs in enumerate(obs_on_road):
            if obs[0] + obs[1] < ts:
                avai_obs = obs

        if ts == 199: # 200 time steps
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
            return None

        if avai_obs == None:
            return None # No action
        #print(avai_obs)
        obs_on_road.remove(avai_obs)
        action = self.choose_actions(avai_obs[2])

        if self.pending_action is not None:
            self.buffer_s.append(self.pending_state)
            self.buffer_a.append(self.pending_action)
            self.buffer_r.append(avai_obs[3])
            #print(avai_obs)
        self.pending_state = avai_obs[2]
        self.pending_action = action

        return action


