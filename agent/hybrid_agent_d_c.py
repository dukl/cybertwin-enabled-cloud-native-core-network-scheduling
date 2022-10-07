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
        self.actor_h_s = []
        self.actor_v_s = []
        self.actor_sch = []
        self.critic = None
        self._build_actor_critic_networks()
        self.a_update_steps = 5
        self.c_update_steps = 5

        self.sess.run(tf.global_variables_initializer())

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

    def choose_actions(self, obs):
        action = ACTIONS()
        act_value = []
        h_scaling = np.zeros((ACS.n_node, len(ACS.t_NFs) - 1))
        v_scaling = [np.zeros(ACS.n_max_inst) for _ in range(ACS.n_node*(len(ACS.t_NFs) - 1))]
        scheduling = [np.zeros(ACS.n_node) for _ in range(len(ACS.msg_msc)*(len(ACS.t_NFs) - 1))]

        thread_list = []

        for i in range(ACS.n_node):
            for j in range(len(ACS.t_NFs) - 1):
                t_ = threading.Thread(target=self.actor_h_scaling(self.actor_h_s[i*(len(ACS.t_NFs)-1) + j], h_scaling, i, j, obs))
                thread_list.append(t_)
                t_.start()
                t_ = threading.Thread(target=self.actor_v_scaling(self.actor_v_s[i*(len(ACS.t_NFs)-1) + j], v_scaling, i, j, obs))
                thread_list.append(t_)
                t_.start()

        for i in range(len(ACS.msg_msc)):
            for j in range(len(ACS.t_NFs) - 1):
                t_ = threading.Thread(target=self.actor_scheduling(self.actor_sch[i*(len(ACS.t_NFs)-1) + j], scheduling, i, j, obs))
                thread_list.append(t_)
                t_.start()

        for j in thread_list:
            j.join()

        print(h_scaling)
        print(v_scaling)
        print(scheduling)

        action.h_s = h_scaling
        action.v_s = v_scaling
        action.sch = scheduling
        return action
