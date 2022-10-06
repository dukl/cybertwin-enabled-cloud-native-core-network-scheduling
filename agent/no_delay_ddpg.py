import random
import numpy, copy
from utils.logger import log
from utils.obs_reward_action_def import ACT
import utils.system_inner as SI
import utils.global_parameters as GP
import environment.compute_reward as CR
import results.running_value as RV
from agent.aiModels.ddpg import DDPG

class NDDDPG:
    def __init__(self):
        self.obs_dim, self.act_dim = SI.CHECK_ACT_OBS_DIM()
        self.ddpg = DDPG(self.obs_dim, self.act_dim)
        self.reqs = [0 for _ in range(len(GP.msc))]
        self.req_index = [0 for _ in range(len(GP.msc))]
        self.pending_obs, self.pending_act = None, None
        self.step_num = 0
        self.train_index = 0

    def reset(self):
        self.reqs = [0 for _ in range(len(GP.msc))]
        self.req_index = [0 for _ in range(len(GP.msc))]
        self.step_num = 0
        self.pending_obs, self.pending_act = None, None

    def receive_requests(self):
        self.reqs = [0 for _ in range(len(GP.msc))]
        for i in range(len(GP.msc)):
            self.reqs[i] = GP.n_reqs_per_msc #random.randint(0,GP.n_reqs_per_msc)
            self.req_index[i] += self.reqs[i]
        return self.reqs

    def _reqs_in_obs(self, reqs):
        obs_reqs = []
        for i in range(len(reqs)):
            req_type = []
            for m in GP.msc[i]:
                req_type.append((m + 1) / len(GP.c_r_ms))
            if len(req_type) < 5:
                req_type += [0 for _ in range(5 - len(req_type))]
            req_type.append(reqs[i] / GP.n_reqs_per_msc)
            obs_reqs += req_type
        return obs_reqs

    def receive_observation_s(self, obs, ts):
        obs_reqs = self._reqs_in_obs(self.receive_requests())
        norm_obs_env = SI.NORM_STATE(copy.deepcopy(obs[0].value))
        obs_input = numpy.array([b for a in norm_obs_env for b in a] + obs_reqs)
        obs_input[obs_input==0] = 0.0001
        #log.logger.debug('agent receive observation (normnized) = (len=%d)\n%s' % (len(obs_input.tolist()),str(obs_input.tolist())))
        action_value = self.ddpg.act(obs_input)[0]
        #action_out   = action_value.reshape(1, self.act_dim)
        action_value = (action_value - numpy.min(action_value))/(numpy.max(action_value) - numpy.min(action_value)) * GP.n_servers*GP.n_ms_server*GP.ypi_max
        action_value = action_value.astype('int')
        #log.logger.debug('action = (len = %d)\n%s' % (len(action_value.tolist()), str(action_value.tolist())))
        valid_action = []
        for i in range(len(self.reqs)): # req types
            for j in range(self.reqs[i]): # req instances
                inter_actions = []
                for k in range(len(GP.msc[i])):
                    idx = i*GP.n_reqs_per_msc*5 + j*5 + k
                    action = action_value[idx]
                    if action == GP.n_servers*GP.n_ms_server*GP.ypi_max:
                        action -= 1
                    server_idx, inst_idx, n_threads = int(action / (GP.n_ms_server * (GP.ypi_max))), int((action % (GP.n_ms_server * (GP.ypi_max))) / (GP.ypi_max)), (action % (GP.n_ms_server * (GP.ypi_max))) % (GP.ypi_max)
                    n_threads += 1
                    #log.logger.debug('req(%d-%d-%d) mapped action=(%d-%d) - (%d-%d-%d)' % (i, j, GP.msc[i][k], idx, action, server_idx, inst_idx, n_threads))
                    inter_actions.append([GP.msc[i][k], server_idx, inst_idx, n_threads, i, j+self.req_index[i]])
                valid_action.append(inter_actions)
        if self.pending_obs is not None:
            self.ddpg.memory.append([self.pending_obs, self.pending_act, obs[0].major_reward, obs_input.reshape(1, obs_input.shape[0])])
        self.pending_obs, self.pending_act = obs_input.reshape(1, obs_input.shape[0]), action_value.reshape(1, action_value.shape[0])
        self.step_num += 1
        batch_size = 32
        if len(self.ddpg.memory) > batch_size and self.step_num % 20 == 0:
            self.train_index += 1
            if self.train_index % 20 == 0:
                log.logger.debug('Replacing...')
                self.ddpg.update_target()
            log.logger.debug('Training...')
            self.ddpg.train(batch_size)
        return ACT(ts, valid_action)
