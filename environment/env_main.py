import utils.global_parameters as GP
from utils.obs_reward_action_def import OBSRWD
from utils.logger import log
import results.running_value as RV
import utils.system_inner as SI
import numpy as np
import math, copy


class NF:
    def __init__(self, type, id):
        self.id = [type, id]
        self.n_threads = 0
        self.lamda = 0
        self.reqs = []
        self.processed_reqs = []
        self.old_lamda = 0
        self.old_n_threads = 0


    def run(self):
        #log.logger.debug('inst(%d-%d) has resources(%d-%d), n_reqs=%d' % (self.id[0], self.id[1], self.lamda, self.n_threads, len(self.reqs)))
        running_time = 0
        index = 0
        if self.n_threads == 0:
            return
        average_time = self.lamda * GP.w_ms[self.id[0]] / self.n_threads + 1/(GP.cpu/GP.psi_ms[self.id[0]]-1)
        one_msg_time = 2*average_time/(self.lamda+1)
        process_time_max = GP.lamda_ms[self.id[0]] * GP.w_ms[self.id[0]] + 1 / (GP.cpu / GP.psi_ms[self.id[0]] - 1)
        one_process_time_max = 2*process_time_max/(GP.lamda_ms[self.id[0]]+1)
        process_time_min = self.lamda * GP.w_ms[self.id[0]] / GP.ypi_max + 1/(GP.cpu/GP.psi_ms[self.id[0]]-1)
        one_process_time_min = 2*process_time_min/(self.lamda + 1)
        #log.logger.debug('one_msg_time = %f' % (one_msg_time))
        while True:
            index += 1
            if len(self.reqs) == 0 or self.n_threads == 0:
                #log.logger.debug('total running time = %f' % (running_time))
                break
            #if index <= self.old_lamda:
            #    lamda = self.old_lamda - index + 1
            #    n_threads = GP.ypi_max
            #    log.logger.debug('using old resources')
            #else:
            #    lamda = self.lamda
            #    n_threads = self.n_threads
            req = self.reqs[0]
            del self.reqs[0]
            #process_time = self.lamda * GP.w_ms[self.id[0]] / self.n_threads + 1/(GP.cpu/GP.psi_ms[self.id[0]]-1)
            process_time = index * one_msg_time
            process_time_max = index * one_process_time_max
            process_time_min = index * one_process_time_min
            running_time += one_msg_time
            self.lamda -= 1
            if running_time > GP.one_step_time:
                #log.logger.debug('total running time = %f, inst(%d-%d) has not been processed req(%d-%d-%d), return' % (running_time, self.id[0], self.id[1], req[0], req[1], self.id[0]))
                self.lamda += 1
                self.reqs.insert(0, req)
                break
            else:
                self.processed_reqs.append(req + [process_time, process_time_max, process_time_min])
            #log.logger.debug('inst(%d-%d) has processed req(%d-%d-%d) for %f' % (self.id[0], self.id[1], req[0], req[1], self.id[0], running_time))
        #self.old_lamda = self.lamda
        #self.old_n_threads = self.n_threads
        #log.logger.debug('inst(%d-%d) resources: lamda=%d, n_threads=%d' % (self.id[0], self.id[1], self.lamda, self.n_threads))
        #log.logger.debug('proposed reqs:')
        #for req in self.processed_reqs:
        #    log.logger.debug('%s' % (str(req)))
        #log.logger.debug('non-processed reqs:')
        #for req in self.reqs:
        #    log.logger.debug('%s' % (str(req)))



class ENV:
    def __init__(self):
        self.nfs = [[] for _ in range(len(GP.c_r_ms))]
        for t in range(len(GP.c_r_ms)):
            for i in range(GP.n_servers):
                for j in range(GP.n_ms_server):
                    self.nfs[t].append(NF(t, i*GP.n_ms_server+j))
        self.left_reqs = []
        self.obs = []

    def reset(self):
        self.obs.clear()
        self.left_reqs.clear()
        for t in range(len(GP.c_r_ms)):
            for i in range(GP.n_servers):
                for j in range(GP.n_ms_server):
                    self.nfs[t][i * GP.n_ms_server + j].lamda = 0
                    self.nfs[t][i * GP.n_ms_server + j].n_threads = 0
                    self.nfs[t][i * GP.n_ms_server + j].reqs.clear()
                    self.nfs[t][i * GP.n_ms_server + j].processed_reqs.clear()

    def send_obs_reward(self, ts):
        self.obs = []
        for t in range(len(GP.c_r_ms)):
            for i in range(GP.n_servers):
                for j in range(GP.n_ms_server):
                    #log.logger.debug('%d' % (t*GP.n_servers*GP.n_ms_server+i*GP.n_ms_server+j))
                    self.obs.append([self.nfs[t][i*GP.n_ms_server+j].lamda, self.nfs[t][i*GP.n_ms_server+j].n_threads])
        log.logger.debug('obs[%d]=\n%s' % (ts, str(self.obs)))
        #log.logger.debug('total reqs left: %d' % (np.sum(np.array(obs)[:,0])))
        return OBSRWD(ts, self.obs, RV.time_step_reward[-1])

    def act(self, action):
        if action is None:
            RV.time_step_reward.append(-1)
            RV.is_start_collect_env = False
            return
        RV.is_start_collect_env = True
        log.logger.debug('before %d reqs, add %d reqs' % (len(self.left_reqs), len(action.value)))
        self.left_reqs += action.value
        #log.logger.debug('env running to process %d reqs' % (len(self.left_reqs)))
        obs_env = copy.deepcopy(self.obs)
        del_index_ele = []
        for req in action.value:
            is_req_mapped_success = True
            last_obs_env_req = copy.deepcopy(obs_env)
            for act in req:
                is_mapped_success = True
                last_obs_env = copy.deepcopy(obs_env)
                [m, s, i, n, req_type, req_id] = act
                #log.logger.debug('%s' % (str(act)))
                idx = m*GP.n_servers*GP.n_ms_server + s*GP.n_ms_server + i
                if SI.CHECK_VALID_ACTION(self.obs, idx, n):
                    self.nfs[m][s * GP.n_ms_server + i].reqs.append([req_type, req_id])
                    self.nfs[m][s * GP.n_ms_server + i].lamda += 1
                    self.nfs[m][s * GP.n_ms_server + i].n_threads += n
                    if self.nfs[m][s * GP.n_ms_server + i].n_threads > GP.ypi_max:
                        self.nfs[m][s * GP.n_ms_server + i].n_threads = GP.ypi_max
                else:
                    is_mapped_success = False
                    is_req_mapped_success = False
                if is_mapped_success is False:
                    obs_env = copy.deepcopy(last_obs_env)
            if is_req_mapped_success is False:
                RV.mapped_succ_rate[-1] += 1
                del_index_ele.append(req)
                obs_env = copy.deepcopy(last_obs_env_req)
        for req in del_index_ele:
            self.left_reqs.remove(req) # un mapped reqs, removing...
        RV.mapped_succ_rate[-1] = (1 - RV.mapped_succ_rate[-1]/len(action.value))

        for m in range(len(GP.c_r_ms)):
            for n in range(GP.n_servers):
                for i in range(GP.n_ms_server):
                    self.nfs[m][n*GP.n_ms_server+i].run()

        index = 0
        Q_t = 0
        del_index_ele = []
        for r in range(len(self.left_reqs)):
            sum = 0
            sum_max = 0
            sum_min = 0
            is_processed = True
            processed_remov_ele = []
            for act in self.left_reqs[r]:
                is_here = False
                [m,s,i,n,req_type, req_id] = act
                #log.logger.debug('searching req(%d-%d) in instance(%d-%d)' % (req_type, req_id, m, s*GP.n_ms_server+i))
                for [r_t, r_i, time, time_max, time_min] in self.nfs[m][s*GP.n_ms_server+i].processed_reqs:
                    if r_t == req_type and r_i == req_id:
                        #log.logger.debug('inst(%d-%d) proposed reqs: %s' % (m, s * GP.n_ms_server + i, str([r_t, r_i])))
                        sum += time
                        sum_max += time_max
                        sum_min += time_min
                        #self.nfs[m][s*GP.n_ms_server+i].processed_reqs.remove([r_t, r_i, time, time_max])
                        processed_remov_ele.append([m,s,i, r_t, r_i, time, time_max, time_min])
                        is_here = True
                        #log.logger.debug('found len(self.nfs.processed_reqs)=%d in instance(%d-%d)' % (len(self.nfs[m][s*GP.n_ms_server+i].processed_reqs), m, s*GP.n_ms_server+i))
                        break
                if is_here is False:
                    is_processed = False
                    #log.logger.debug('searching req(%d-%d) in instance(%d-%d) not found'%(req_type,req_id, m, s*GP.n_ms_server+i))
                    break
            if is_processed is False:
                #log.logger.debug('searching req(%d-%d) failed' % (req_type, req_id))
                #break
                pass
            else:
                for [m,s,i, r_t, r_i, time, time_max, time_min] in processed_remov_ele:
                    self.nfs[m][s*GP.n_ms_server+i].processed_reqs.remove([r_t, r_i, time, time_max, time_min])
                index += 1
                del_index_ele.append(self.left_reqs[r])
                #log.logger.debug('searching req(%d-%d) successfully' % (req_type, req_id))
                Q_t += (sum_max - sum)/(sum_max - sum_min)
        if len(self.left_reqs) == 0:
            Q_t = 0
        else:
            Q_t = Q_t/len(self.left_reqs)
        log.logger.debug('processed %d reqs from %d reqs' % (index, len(self.left_reqs)))


        sum_threads = 0
        for m in range(len(GP.c_r_ms)):
            for n in range(GP.n_servers):
                for i in range(GP.n_ms_server):
                    sum_threads += self.nfs[m][n*GP.n_ms_server+i].n_threads

        log.logger.debug('sum_threads = %d, max_threads = %d, resource_rate = %f' % (sum_threads, GP.n_ms_server*GP.n_servers*GP.ypi_max*len(GP.c_r_ms), (1-sum_threads/(GP.n_ms_server*GP.n_servers*GP.ypi_max*len(GP.c_r_ms)))))
        resource_rate = 1 - (sum_threads / (GP.n_ms_server*GP.n_servers*GP.ypi_max*len(GP.c_r_ms)))
        #major_reward *= resource_rate
        log.logger.debug('detailed-reward: succ_rate=%f, Q_t=%f, total_req=%d, resource_rate=%f' % (RV.mapped_succ_rate[-1], Q_t, len(action.value), resource_rate))
        #log.logger.debug('time step major reward = %f' % (major_reward))
        #major_reward = major_reward * GP.beta_r + GP.beta_r
        #major_reward = (resource_rate * 10 + 10)
        #major_reward = (Q_t * 10)
        major_reward = (RV.mapped_succ_rate[-1] * 50)
        #major_reward /= (10+20)
        log.logger.debug('(scale) time step major reward: %f+%f+%f = %f' % (0, (Q_t * 10), (RV.mapped_succ_rate[-1] * 50), major_reward))
        #major_reward = math.exp(2*Q_t) + 20
        #major_reward += math.exp(resource_rate) + 10
        #major_reward += (2 - math.exp(-2*action.n_mapped_succ_rate)) + 30
        #log.logger.debug('(scale) time step major reward: %f+%f+%f = %f' % ((2 - math.exp(-2*action.n_mapped_succ_rate)) + 30, math.exp(2*Q_t) + 20, math.exp(resource_rate) + 10, major_reward))
        RV.time_step_reward.append(major_reward)
        RV.episode_reward[-1] += major_reward

        for req in del_index_ele:
            self.left_reqs.remove(req) # successfully processed reqs, removing...

        deleted_reqs = del_index_ele
        for req in deleted_reqs:
            for act in req:
                [m, s, i, n, req_type, req_id] = act
                ##log.logger.debug('minus [%d,%d,%d].n_threads = %d' % (m,s,i,n))
                self.nfs[m][s * GP.n_ms_server + i].n_threads -= n
                if self.nfs[m][s * GP.n_ms_server + i].n_threads <= 0:
                    if self.nfs[m][s * GP.n_ms_server + i].lamda > 0:
                        self.nfs[m][s * GP.n_ms_server + i].n_threads = GP.ypi_max
                    else:
                        self.nfs[m][s * GP.n_ms_server + i].n_threads = 0
                #self.nfs[m][s * GP.n_ms_server + i].lamda -= 1


