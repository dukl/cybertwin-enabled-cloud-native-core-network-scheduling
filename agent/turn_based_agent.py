import utils.global_parameters as GP
import results.running_value as RV
import utils.system_inner as SI
import random, copy, numpy
import environment.compute_reward as CR
from utils.logger import log
from utils.obs_reward_action_def import ACT
from agent.aiModels.dqn_agents import DDQNAgent

class TBA:
    def __init__(self):
        self.reqs = [0 for _ in range(len(GP.arrive_rate))]
        self.obs_dim, self.act_dim = SI.CHECK_ACT_OBS_DIM()
        self.model = DDQNAgent(self.obs_dim, self.act_dim, False, False, 0, 0.001, 0.999, 0.0001, 1, False, True)
        self.step_num = 0

    def reset(self):
        pass

    def receive_requests(self):
        for i in range(len(GP.msc)):
            loc = random.randint(0,9)
            self.reqs[i] += int(GP.arrive_rate[loc])

    def compute_action(self, is_random, ts, obs_env):
        valid_action = []
        tmp_memory = []
        for i in range(len(GP.msc)):
            for j in range(self.reqs[i]):
                is_req_mapped_success = True
                inter_actions = []
                last_obs_env_req = copy.deepcopy(obs_env)
                for ms in GP.msc[i]:
                    is_mapped_success = True
                    last_obs_env = copy.deepcopy(obs_env)
                    obs_req = [i, self.reqs[i], ms]
                    norm_obs_env = SI.NORM_STATE(copy.deepcopy(obs_env))
                    obs_input = numpy.array([b for a in norm_obs_env for b in a] + obs_req)
                    if is_random is True:
                        action = random.randint(0, GP.n_ms_server * GP.n_servers * (GP.ypi_max + 1) - 1)
                    else:
                        action = self.model.act(obs_input, eval=False)
                    server_idx, inst_idx, n_threads = int(action / (GP.n_ms_server * (GP.ypi_max + 1))), int((action % (GP.n_ms_server * (GP.ypi_max + 1))) / (GP.ypi_max + 1)), (action % (GP.n_ms_server * (GP.ypi_max + 1))) % (GP.ypi_max + 1)
                    # log.logger.debug('action=%d -> server_idx=%d, inst_idx=%d, n_threads=%d' % (action, server_idx, inst_idx, n_threads))
                    idx = ms * GP.n_servers * GP.n_ms_server + server_idx * GP.n_ms_server + inst_idx
                    # log.logger.debug('trying to map req=(%d,%d,%d) into instance=(%d,%d,%d) n_threads=%d' % (i, j, ms, ms, server_idx, inst_idx, n_threads))
                    if SI.CHECK_VALID_ACTION(obs_env, idx, n_threads):
                        inter_actions.append([ms, server_idx, inst_idx, n_threads])
                    else:
                        is_mapped_success = False
                        is_req_mapped_success = False
                        # log.logger.debug('punish action = %d' % (action))
                    obs_env[idx][0] += 1
                    obs_env[idx][1] += n_threads
                    if obs_env[idx][1] > GP.ypi_max:
                        obs_env[idx][1] = GP.ypi_max
                    norm_obs_env_next = SI.NORM_STATE(copy.deepcopy(obs_env))
                    obs_next = numpy.array([b for a in norm_obs_env_next for b in a] + obs_req)
                    # obs_next[obs_next==0] = 0.0001
                    minor_reward = CR.compute_minor_reward(is_mapped_success, GP.msc[i].index(ms), self.reqs, i, obs_env[idx], ms, n_threads)
                    # log.logger.debug('minor reward = %f' % (minor_reward))
                    tmp_memory.append([obs_input, action, minor_reward, obs_next])

                    if is_mapped_success is False:
                        obs_env = copy.deepcopy(last_obs_env)
                if is_req_mapped_success is False:
                    RV.mapped_succ_rate[-1] += 1
                    # log.logger.debug('req is mapped unsuccessfully')
                    obs_env = copy.deepcopy(last_obs_env_req)
                else:
                    # log.logger.debug('req is mapped successfully')
                    valid_action.append(inter_actions)
        if ts >= 0:
            RV.memory.append(tmp_memory)
        n_successful_mapped_reqs = sum(self.reqs) - RV.mapped_succ_rate[-1]
        RV.mapped_succ_rate[-1] = 1 - RV.mapped_succ_rate[-1] / sum(self.reqs)
        log.logger.debug('t=%d, successful mapped rate = %f, n_successful_mapped_reqs = %d' % (ts, RV.mapped_succ_rate[-1], n_successful_mapped_reqs))
        ##log.logger.debug('agent-obs[%d]=\n%s' % (ts+1, str(obs_env)))
        self.reqs = [0 for _ in range(len(GP.arrive_rate))]
        return ACT(ts, valid_action, RV.mapped_succ_rate[-1], n_successful_mapped_reqs)

    def receive_observation_s(self, obs, ts):
        self.receive_requests()
        if len(obs) == 0:
            log.logger.debug('Agent doesn\'t receive any obs at time %d' % (ts))
            RV.memory.append(None)
            return None
        for o in obs:
            if o.major_reward != -1 and RV.memory[o.id-1] is not None:
                for mem in RV.memory[o.id - 1]:
                    mem[2] += o.major_reward
                    self.model.memorize(mem[0], mem[1], mem[2], mem[3], False)
        self.step_num += 1
        if self.step_num % 200 == 0:
            log.logger.debug('Replacing...')
            self.model.update_target_model()
        batch_size = 32
        if len(self.model.memory) > batch_size and self.step_num % 10 == 0:
            log.logger.debug('Training...')
            batch_loss_dict = self.model.replay(batch_size)
        log.logger.debug('Agent receives %d obs at time %d' % (len(obs), ts))
        obs.sort(key=lambda OBSRWD: OBSRWD.id, reverse=True)
        obs_env = obs[0].value
        action = self.compute_action(False, ts, copy.deepcopy(obs_env))
        return action

