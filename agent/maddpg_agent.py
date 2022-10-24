import numpy as np
import tensorflow as tf

from agent.aiModels.maddpg_new import MADDPG
from utils.actions_definition import ACTIONS
import utils.auto_scaling_settings as ACS
import random

class MADDPG_NEW:
    def __init__(self):
        self.sess = tf.Session()

        self.actor_h_s = MADDPG('a_h_s', nb_actions=ACS.actors[0][0], nb_input=ACS.n_node * 8, nb_other_aciton=ACS.actors[1][0], actor_id=0)
        self.targ_actor_h_s = MADDPG('targ_a_h_s', nb_actions=ACS.actors[0][0], nb_input=ACS.n_node * 8, nb_other_aciton=ACS.actors[1][0], actor_id=0)

        self.actor_v_s = MADDPG('a_v_s', nb_actions=ACS.actors[1][0], nb_input=ACS.n_node * 8, nb_other_aciton=ACS.actors[0][0], actor_id=1)
        self.targ_actor_v_s = MADDPG('targ_a_v_s', nb_actions=ACS.actors[1][0], nb_input=ACS.n_node * 8, nb_other_aciton=ACS.actors[0][0], actor_id=1)

        #self.actor_sch = MADDPG('a_sch', nb_actions=ACS.actors[2][0], nb_input=ACS.n_node * 8, nb_other_aciton=ACS.actors[0][0]+ACS.actors[1][0], actor_id=2)
        #self.targ_actor_sch = MADDPG('targ_a_sch', nb_actions=ACS.actors[2][0], nb_input=ACS.n_node * 8, nb_other_aciton=ACS.actors[0][0]+ACS.actors[1][0], actor_id=2)

        self.agent1_actor_target_init, self.agent1_actor_target_update = self.create_init_update('agent1_actor', 'agent1_target_actor')
        self.agent1_critic_target_init, self.agent1_critic_target_update = self.create_init_update('agent1_critic',
                                                                                    'agent1_target_critic')

        self.agent2_actor_target_init, self.agent2_actor_target_update = self.create_init_update('agent2_actor', 'agent2_target_actor')
        self.agent2_critic_target_init, self.agent2_critic_target_update = self.create_init_update('agent2_critic',
                                                                                    'agent2_target_critic')

        self.agent3_actor_target_init, self.agent3_actor_target_update = self.create_init_update('agent3_actor', 'agent3_target_actor')
        self.agent3_critic_target_init, self.agent3_critic_target_update = self.create_init_update('agent3_critic',
                                                                                    'agent3_target_critic')

        self.sess.run(tf.global_variables_initializer())
        self.sess.run([self.agent1_actor_target_init, self.agent1_critic_target_init,
              self.agent2_actor_target_init, self.agent2_critic_target_init,
              self.agent3_actor_target_init, self.agent3_critic_target_init])

        #self.actor_h_s_memory = ReplayBuffer(100000)
        #self.actor_v_s_memory = ReplayBuffer(100000)
        #self.actor_sch_memory = ReplayBuffer(100000)
        self.actor_h_s_memory = []
        self.actor_v_s_memory = []
        self.actor_sch_memory = []

        self.update_it = 0
        self.pending_s, self.pending_a = None, None
        self.last_obs_id = -1

    def reset(self):
        self.pending_s, self.pending_a = None, None
        self.last_obs_id = -1


    def create_init_update(self, exec_name, target_name, tau=0.99):
        online_var = [i for i in tf.trainable_variables() if exec_name in i.name]
        target_var = [i for i in tf.trainable_variables() if target_name in i.name]

        target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
        target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in
                         zip(online_var, target_var)]

        return target_init, target_update

    def get_agents_action(self, o_n, sess, noise_rate=0.0):
        #print(o_n.shape)
        agent1_action = self.actor_h_s.action(state=[o_n], sess=sess) + np.random.randn(ACS.actors[0][0]) * noise_rate
        agent2_action = self.actor_v_s.action(state=[o_n], sess=sess) + np.random.randn(ACS.actors[1][0]) * noise_rate
        #agent3_action = self.actor_sch.action(state=[o_n], sess=sess) + np.random.randn(ACS.actors[2][0]) * noise_rate
        #print(agent1_action)
        #print(agent2_action)
        #print(agent3_action)
        return agent1_action, agent2_action, None

    def choose_actions(self, obs):
        action = ACTIONS()
        h_s_out, v_s_out, scheduling = self.get_agents_action(obs, self.sess, noise_rate=0.2)
        action.raw_h_s = h_s_out
        action.raw_v_s = v_s_out
        #action.raw_sch = scheduling

        h_s_out += 1e-5
        h_s_out = (np.max(h_s_out) - h_s_out) / (np.max(h_s_out) - np.min(h_s_out))
        h_s_out[h_s_out < 0.33] = -1
        h_s_out[h_s_out > 0.66] = 1

        h_s_out = h_s_out.reshape((ACS.n_node, len(ACS.t_NFs) - 1))
        #print(h_s_out.astype('int32'))
        action.h_s = h_s_out.astype('int32')

        #v_s_out = v_s_out/np.sum(v_s_out)
        v_s = v_s_out + 1e-5
        v_s = (np.max(v_s) - v_s) / (np.max(v_s) - np.min(v_s))
        v_s[v_s < 0.33] = -1
        v_s[v_s > 0.66] = 1
        v_s = v_s.reshape((ACS.n_node * (len(ACS.t_NFs) - 1), ACS.n_max_inst))

        #v_s = v_s.reshape((ACS.n_node*(len(ACS.t_NFs) - 1), ACS.n_max_inst))
        #for i in range(v_s.shape[0]):
        #    v_s[i, :] = v_s[i, :] / np.sum(v_s[i, :])
        #v_s_out = v_s_out.reshape((ACS.n_node * (len(ACS.t_NFs) - 1), ACS.n_max_inst))
        #print(v_s.astype('int32'))
        action.v_s = v_s.astype('int32')


        #scheduling += 1e-5
        #scheduling = (np.max(scheduling) - scheduling) / (np.max(scheduling) - np.min(scheduling))

        #scheduling = scheduling.reshape((len(ACS.msg_msc) * (len(ACS.t_NFs) - 1), ACS.n_node))
        #print(scheduling)
        #action.sch = scheduling

        return action

    def train_agent(self, agent_ddpg, agent_ddpg_target, agent_actor_target_update, agent_critic_target_update, other_actors, batch_size, memory):
        s, a1, a2, a3, r, s_ = [], [], [], [], [], []
        samples = np.array(random.sample(memory, batch_size))
        for sample in samples:
            s.append(sample[0])
            a1.append(sample[1])
            a2.append(sample[2])
            #a3.append(sample[3])
            r.append(sample[3])
            s_.append(sample[4])
        s, a1, a2, r, s_ = np.vstack(s), np.vstack(a1), np.vstack(a2), np.vstack(r), np.vstack(s_)
        next_other_action = np.hstack([other_actors[0].action(s_, self.sess)])
        target = r.reshape(-1, 1) + 0.9999 * agent_ddpg_target.Q(state=s_, action=agent_ddpg_target.action(s_, self.sess), other_action=next_other_action, sess=self.sess)
        other_act_batch = np.hstack([a2])
        agent_ddpg.train_actor(state=s, other_action=other_act_batch, sess=self.sess)
        agent_ddpg.train_critic(state=s, action=a1, other_action=other_act_batch, target=target, sess=self.sess)

        if self.update_it >= 10 and self.update_it % 10 == 0:
            self.sess.run([agent_actor_target_update, agent_critic_target_update])


    def memorize(self, s, a, r, s_):
        batch_size = 100
        a_h_s = a.raw_h_s
        a_v_s = a.raw_v_s
        #a_sch = a.raw_sch
        self.actor_h_s_memory.append([s,a_h_s,a_v_s,r,s_])
        self.actor_v_s_memory.append([s,a_v_s,a_h_s,r,s_])
        #self.actor_sch_memory.append([s,a_sch,a_h_s,a_v_s,r,s_])

        if len(self.actor_h_s_memory) > 100 and len(self.actor_h_s_memory) % batch_size == 0:
            print('trainning')
            self.update_it += 1
            self.train_agent(self.actor_h_s, self.targ_actor_h_s, self.agent1_actor_target_update, self.agent1_critic_target_update, [self.targ_actor_v_s], batch_size, memory=self.actor_h_s_memory)
            self.train_agent(self.actor_v_s, self.targ_actor_v_s, self.agent2_actor_target_update, self.agent2_critic_target_update, [self.targ_actor_h_s], batch_size, memory=self.actor_v_s_memory)
            #self.train_agent(self.actor_sch, self.targ_actor_sch, self.agent3_actor_target_update, self.agent3_critic_target_update, [self.targ_actor_h_s, self.targ_actor_v_s], batch_size, memory=self.actor_sch_memory)

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

        action = self.choose_actions(avai_obs[2])

        if self.pending_s is not None:
            self.memorize(self.pending_s, self.pending_a, avai_obs[3], avai_obs[2])
        self.pending_s, self.pending_a = avai_obs[2], action

        return action