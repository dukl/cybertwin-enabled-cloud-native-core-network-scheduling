from utils.obs_reward_action_def import OBSRWD
import results.running_value as RV
import utils.global_parameters as GP
from utils.logger import log

def CHECK_OBSERVATIONS(ts):
    O = []
    for ob in RV.obs_on_road:
        if ob.id + ob.obs_delay <= ts:
            O.append(ob)
            RV.obs_on_road.remove(ob)
    return O

def CHECK_VALID_ACTION(obs_env, index, n_threads):

    for n in range(GP.n_servers):
        sum = 0
        for m in range(len(GP.c_r_ms)):
            for i in range(GP.n_ms_server):
                idx = m*GP.n_servers*GP.n_ms_server + n*GP.n_ms_server + i
                if index == idx:
                    #log.logger.debug('changing [m,s,i,n] = [%d,%d,%d,%d]' % (m, n, i, n_threads))
                    if obs_env[idx][1] + n_threads == 0:
                        #log.logger.debug('final threads = 0: [%d,%d,%d,%d]' % (m, n, i, n_threads))
                        return False
                    if obs_env[idx][0] + 1 > GP.lamda_ms[m]:
                        #log.logger.debug('lamda = %d' % (obs_env[idx][0]))
                        #log.logger.debug('lamda exceed maximum lamda of microservice (%d=%d): [%d,%d,%d,%d]' % (m, GP.lamda_ms[m], m, n, i, n_threads))
                        return False
                    if obs_env[idx][1] + n_threads > GP.ypi_max:
                        #log.logger.debug('exceed maximum threads: [%d,%d,%d,%d]' % (m, n, i, n_threads))
                        #return False
                        sum += (GP.c_r_ms[m] + GP.psi_ms[m]*(GP.ypi_max))
                        continue
                    if obs_env[idx][1] + n_threads > 0:
                        sum += (GP.c_r_ms[m] + GP.psi_ms[m]*(obs_env[idx][1]+n_threads))
                else:
                    if obs_env[idx][1] > 0:
                        sum += (GP.c_r_ms[m] + GP.psi_ms[m]*obs_env[idx][1])
        #log.logger.debug('server[%d] total CPU = %d' % (n, sum))
        if sum > GP.n_cpu_core*GP.cpu:
            #log.logger.debug('exceed maximum cpu')
            return False

    return True

def CHECK_ACT_OBS_DIM():
    return GP.n_servers*GP.n_ms_server*len(GP.c_r_ms)*2 + 3, GP.n_ms_server*GP.n_servers*(GP.ypi_max+1)

def RESET(env, agent):
    env.reset()
    agent.reset()
    RV.obs_on_road.clear()
    RV.time_step_reward = [-1]
    RV.mapped_succ_rate.clear()
    RV.memory.clear()

def NORM_STATE(obs_env):
    for m in range(len(GP.c_r_ms)):
        for n in range(GP.n_servers):
            for i in range(GP.n_ms_server):
                idx = m*GP.n_servers*GP.n_ms_server + n*GP.n_ms_server + i
                #log.logger.debug('lamda=%f, n_threads=%f' % (obs_env[idx][0], obs_env[idx][1]))
                obs_env[idx][0] /= GP.lamda_ms[m]
                obs_env[idx][1] /= GP.ypi_max
    return obs_env