import random
import numpy
from utils.logger import log
from utils.obs_reward_action_def import ACT
import utils.system_inner as SI
import utils.global_parameters as GP
import environment.compute_reward as CR

class NDDQN:
    def __init__(self):
        pass
    def reset(self):
        pass

    def receive_requests(self):
        reqs = [0 for _ in range(len(GP.arrive_rate))]
        for i in range(len(GP.msc)):
            loc = random.randint(0,9)
            reqs[i] = int(GP.arrive_rate[loc])
        return reqs

    def receive_observation_s(self, obs, ts):
        inter_actions = []
        reqs = self.receive_requests()
        obs_env = obs[0].value
        for i in range(len(GP.msc)):
            for j in range(reqs[i]):
                for ms in GP.msc[i]:
                    is_mapped_success = True
                    last_obs_env = obs_env.copy()
                    obs_req = [i, reqs[i], ms]
                    obs_input = numpy.array([b for a in obs_env for b in a] + obs_req)
                    #log.logger.debug('obs_input=\n%s' % (str(obs_input)))
                    action = random.randint(0, GP.n_ms_server*GP.n_servers*GP.ypi_max)
                    server_idx, inst_idx, n_threads = int(action/(GP.n_ms_server*(GP.ypi_max+1))), int((action%(GP.n_ms_server*(GP.ypi_max+1)))/GP.ypi_max), (action%(GP.n_ms_server*(GP.ypi_max+1)))%GP.ypi_max
                    #log.logger.debug('action=%d -> server_idx=%d, inst_idx=%d, n_threads=%d' % (action, server_idx, inst_idx, n_threads))
                    idx = ms*GP.n_servers*GP.n_ms_server + server_idx*GP.n_servers + inst_idx
                    obs_env[idx][0] += 1
                    obs_env[idx][1] += n_threads
                    if SI.CHECK_VALID_ACTION(obs_env, idx, n_threads):
                        inter_actions.append([ms, server_idx, inst_idx, n_threads])
                        log.logger.debug('valid action')
                    else:
                        is_mapped_success = False
                        log.logger.debug('punish action = %d' % (action))
                    reward = CR.compute_minor_reward(is_mapped_success, GP.msc[i].index(ms), reqs, i)
                    log.logger.debug('reward = %f' % (reward))
                    if ~is_mapped_success:
                        obs_env = last_obs_env
        #log.logger.debug('agent-obs[%d]=\n%s' % (ts+1, str(obs_env)))
        return ACT(ts, inter_actions)


