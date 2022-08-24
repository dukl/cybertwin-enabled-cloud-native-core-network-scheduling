import utils.global_parameters as GP
import random, copy, numpy
from utils.obs_reward_action_def import OBSRWD, ACT
import numpy as np
import utils.system_inner as SI
import environment.compute_reward as CR
import results.running_value as RV
from utils.logger import log
from agent.aiModels.dqn_agents import DDQNAgent
from agent.aiModels.forward_model import FM

class DDQNMLP():
    def __init__(self, delay):
        self.obs_dim, self.act_dim = SI.CHECK_ACT_OBS_DIM()
        self.delay = delay
        self.A, self.A_avai = [], []  # RODC-DDPG line-5
        self.sc = OBSRWD(-1, [[0,0] for _ in range(GP.n_ms_server*GP.n_servers*len(GP.c_r_ms))],None)  # latest observation RODC-DDPG Line-6
        self.T = []
        self.D = []
        #self.model = DDQNPlanningAgent(self.obs_dim, self.act_dim, False, False, 1, 0.001, 0.999, 0.001, 0.01, False, True, None, True)
        self.forward_model = FM()
        self.model = DDQNAgent(self.obs_dim, self.act_dim, False, False, 0, 0.001, 0.999, 0.001, 1, False, True)
        self.step_num = 0

    def reset(self):
        pass

    def receive_requests(self):
        reqs = [0 for _ in range(len(GP.arrive_rate))]
        for i in range(len(GP.msc)):
            loc = random.randint(0,9)
            reqs[i] = int(GP.arrive_rate[loc])
        return reqs

    def compute_action(self, is_random, ts, obs_env):
        reqs = self.receive_requests()
        valid_action = []
        tmp_memory = []
        for i in range(len(GP.msc)):
            for j in range(reqs[i]):
                is_req_mapped_success = True
                inter_actions = []
                last_obs_env_req = copy.deepcopy(obs_env)
                for ms in GP.msc[i]:
                    is_mapped_success = True
                    last_obs_env = copy.deepcopy(obs_env)
                    obs_req = [i, reqs[i], ms]
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
                    minor_reward = CR.compute_minor_reward(is_mapped_success, GP.msc[i].index(ms), reqs, i, obs_env[idx], ms, n_threads)
                    # log.logger.debug('minor reward = %f' % (minor_reward))
                    tmp_memory.append([obs_input, action, minor_reward, obs_next])
                    self.forward_model.memory.append([obs_input[0:-3], action, obs_next[0:-3]])

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
        n_successful_mapped_reqs = sum(reqs) - RV.mapped_succ_rate[-1]
        RV.mapped_succ_rate[-1] = 1 - RV.mapped_succ_rate[-1] / sum(reqs)
        log.logger.debug('t=%d, successful mapped rate = %f, n_successful_mapped_reqs = %d' % (ts, RV.mapped_succ_rate[-1], n_successful_mapped_reqs))
        ##log.logger.debug('agent-obs[%d]=\n%s' % (ts+1, str(obs_env)))
        return ACT(ts, valid_action, RV.mapped_succ_rate[-1], n_successful_mapped_reqs)


    def receive_observation_s(self, obs, ts):
        self.step_num += 1
        if self.step_num % 10 == 0:
            log.logger.debug('Training Forward Model...')
            self.forward_model.train(32)
        if self.step_num % 200 == 0:
            log.logger.debug('Replacing DQN Target Network...')
            self.model.update_target_model()
        if len(self.model.memory) > 32 and self.step_num % 20 == 0:
            log.logger.debug('Training DQN Model...')
            batch_loss_dict = self.model.replay(32)


        log.logger.debug('[line-15][Check: id(sc)=%d, len(A)=%d]' % (self.sc.id, len(self.A)))
        if self.sc.id is -1 and len(self.A) is 0:
            action = self.compute_action(True, -1, copy.deepcopy(self.sc.value))
            self.A.append(action)
            self.A_avai.append(action)
            log.logger.debug('[line-16][initialize an action a[%d], len(A)=%d, len(A_avai)=%d]' % (action.id,len(self.A),len(self.A_avai)))
        else:
            self.T.extend(obs)
            log.logger.debug('[line-18][agent adds observations and rewards into T: len(T)=%d]' % (len(self.T)))
            if len(obs) is not 0:
                O_avai = []
                log.logger.debug('[line-7][Initialize the received observations O_avai: len(O_avai)=%d]' % (len(O_avai)))
                for ob in obs:
                    if ob.id > self.sc.id:
                        O_avai.append(ob)
                log.logger.debug('[line-19][Check O_avai: len(O_avai)=%d]' % (len(O_avai)))
                if len(O_avai) is not 0:
                    O_avai.sort(key=lambda OBSRWD: OBSRWD.id, reverse=True)
                    self.sc = O_avai[0]
                    log.logger.debug('[line-20][agent reset sc = s[%d]]' % (self.sc.id))
            self.A_avai.clear()
            for act in self.A:
                if act.id >= self.sc.id and act.id < ts:
                    self.A_avai.append(act)
            log.logger.debug('[line-22][agent reset A_avai: len(A_avai)=%d/ len(A)=%d]' % (len(self.A_avai),len(self.A)))
        log.logger.debug('[agent reset sc to be s[%d] at time step %d]' % (self.sc.id, ts))
        for a in self.A_avai:
            log.logger.debug('[agent reset A_avai to be a[%d] at time step %d]' % (a.id, ts))
        pred_obs_env = self.forward_model.predict(self.sc.value, self.A_avai, ts)
        action = self.compute_action(False, ts, copy.deepcopy(pred_obs_env))
        self.A.append(action)
        log.logger.debug('[line-25][agent adds a[%d] into A: len(A)=%d, len(A_avai)=%d]' % (action.id, len(self.A), len(self.A_avai)))
        log.logger.debug('[line-27][Check if exists adjacent obs]')
        for robs in self.T:
            if robs.id == 0:
                continue
            for mem in RV.memory[robs.id - 1]:
                mem[2] += robs.major_reward
                self.model.memorize(mem[0], mem[1], mem[2], mem[3], False)
        self.T.clear()
                    
        return action