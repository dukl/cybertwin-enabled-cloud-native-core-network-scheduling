import random

import keras.models
import numpy as np


from utils.logger import log
import utils.auto_scaling_settings as ACS
from agent.aiModels.dqn_agents import DDQNPlanningAgent, DDQNAgent

class Agent:
    def __init__(self):
        self.action_space = [-1, 0, 1]
        self.history_obs = []
        self.history_acts = []
        self.history_rwds = []
        #self.model = DQN(
        #    3, 25, learning_rate=0.001, reward_decay=0.9, e_greedy=0.9, replace_target_iter=20, memory_size=100000, batch_size=1280
        #)
        #self.model = DDQNPlanningAgent(25, 3, False, False, 1, 0.001, 0.999, 0.001, 0.01, False, True, None, True)
        #self.model = DDQNAgent(27, 3, False, False, 0, 0.001, 0.999, 0.001, 1, False, True)
        self.model = DDQNAgent(31, 3, False, False, 0, 0.001, 0.99996, 0.0001, 0.9, False, True)
        self.step = 0
        self.pending_state = None
        self.pending_state_next = None
        self.pending_action = None
        self.epison_reward = []
        self.index = 0
        self.reward_sum = 0
        self.isPredGT = False
        self.isPredDNN = False
        self.act_buf = []
        self.obs_buf = []
        self.step_num = 0
        self.last_obs_id = -1

    def reset(self):
        self.last_obs_id = -1
        self.step_num = 0
        self.step = 0
        self.pending_state = None
        self.pending_action = None
        self.index = 0
        self.reward_sum = 0
        self.isPredGT = False
        self.isPredDNN = False
        self.act_buf.clear()
        self.obs_buf.clear()
        self.model.clear_action_buffer()

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

        action = self.model.act(avai_obs[2], eval=False)

        if self.pending_state is not None:
            self.model.memorize(self.pending_state, self.pending_action, avai_obs[3], avai_obs[2], False)
        self.pending_state, self.pending_action = avai_obs[2], action

        self.step_num += 1
        if self.step_num % 200 == 0:
            self.model.update_target_model()
        batch_size = 32
        if len(self.model.memory) > batch_size and self.step_num % 10 == 0:
            batch_loss_dict = self.model.replay(batch_size)

        return action