import utils.global_parameters as GP
from utils.obs_reward_action_def import OBSRWD
from utils.logger import log
import results.running_value as RV


class NF:
    def __init__(self, type, id):
        self.id = [type, id]
        self.n_threads = 0
        self.lamda = 0


class ENV:
    def __init__(self):
        self.nfs = [[] for _ in range(len(GP.c_r_ms))]
        for t in range(len(GP.c_r_ms)):
            for i in range(GP.n_servers):
                for j in range(GP.n_ms_server):
                    self.nfs[t].append(NF(t, i*GP.n_ms_server+j))
        self.left_reqs = []

    def reset(self):
        pass
    def send_obs_reward(self, ts):
        obs = []
        for t in range(len(GP.c_r_ms)):
            for i in range(GP.n_servers):
                for j in range(GP.n_ms_server):
                    obs.append([self.nfs[t][i*GP.n_ms_server+j].lamda, self.nfs[t][i*GP.n_ms_server+j].n_threads])
        log.logger.debug('obs[%d]=\n%s' % (ts, str(obs)))
        return OBSRWD(ts, obs, RV.time_step_reward[-1])

    def act(self, action):
        log.logger.debug('before %d reqs, add %d reqs' % (len(self.left_reqs), len(action.value)))
        self.left_reqs += action.value
        log.logger.debug('env running to process %d reqs' % (len(self.left_reqs)))
        for req in action.value:
            for act in req:
                [m, s, i, n] = act
                log.logger.debug('[m,s,i,n]=[%d,%d,%d,%d]' % (m,s,i,n))
                self.nfs[m][s * GP.n_ms_server + i].lamda += 1
                self.nfs[m][s * GP.n_ms_server + i].n_threads += n
        total_time = 0
        index = 0
        Q_t = 0
        for r in range(len(self.left_reqs)):
            sum = 0
            sum_max = 0
            sum_min = 0
            for act in self.left_reqs[r]:
                [m, s, i, n] = act
                if self.nfs[m][s * GP.n_ms_server + i].n_threads == 0:
                    self.nfs[m][s * GP.n_ms_server + i].n_threads = 1
                log.logger.debug('[%d,%d,%d].lamda = %d' % (m,s,i, self.nfs[m][s * GP.n_ms_server + i].lamda))
                log.logger.debug('[%d,%d,%d].n_threads = %d' % (m, s, i, self.nfs[m][s * GP.n_ms_server + i].n_threads))
                sum += self.nfs[m][s * GP.n_ms_server + i].lamda * GP.w_ms[m] / self.nfs[m][s * GP.n_ms_server + i].n_threads + 1/(GP.cpu/GP.psi_ms[m] - 1)
                sum_max += GP.lamda_ms[m]*GP.w_ms[m] / 1 + 1/(GP.cpu/GP.psi_ms[m] - 1)
                sum_min += self.nfs[m][s * GP.n_ms_server + i].lamda * GP.w_ms[m] / GP.ypi_max + 1/(GP.cpu/GP.psi_ms[m] - 1)
                self.nfs[m][s * GP.n_ms_server + i].lamda -= 1
            log.logger.debug('response time = %f, max = %f, min = %f, q_t = %f' % (sum, sum_max, sum_min, (sum_max - sum)/(sum_max - sum_min)))
            Q_t += (sum_max - sum)/(sum_max - sum_min)
            total_time += sum
            log.logger.debug('total response time = %f' % (total_time))
            if total_time > GP.one_step_time:
                index = r + 1
                break
        if total_time <= GP.one_step_time:
            index = r + 1
        log.logger.debug('processed %d reqs from %d reqs' % (index, len(self.left_reqs)))
        Q_t = Q_t/(len(self.left_reqs))
        log.logger.debug('Q_t = %f' % (Q_t))
        major_reward = action.n_mapped_succ_rate * Q_t*action.total_reqs*GP.beta_r
        log.logger.debug('major reward-1 = %f' % (major_reward))

        sum_threads = 0
        for m in range(len(GP.c_r_ms)):
            for n in range(GP.n_servers):
                for i in range(GP.n_ms_server):
                    sum_threads += self.nfs[m][n*GP.n_ms_server+i].n_threads

        log.logger.debug('sum_threads = %d, max_threads = %d, resource_rate = %f' % (sum_threads, GP.n_ms_server*GP.n_servers*GP.ypi_max*len(GP.c_r_ms), (1-sum_threads/(GP.n_ms_server*GP.n_servers*GP.ypi_max*len(GP.c_r_ms)))))

        major_reward = (1 - (sum_threads / (GP.n_ms_server*GP.n_servers*GP.ypi_max*len(GP.c_r_ms)))) * major_reward
        log.logger.debug('major reward = %f' % (major_reward))
        RV.time_step_reward.append(major_reward)

        deleted_reqs = self.left_reqs[0:index]
        del self.left_reqs[0:index]
        for req in deleted_reqs:
            for act in req:
                [m, s, i, n] = act
                #log.logger.debug('minus [%d,%d,%d].n_threads = %d' % (m,s,i,n))
                self.nfs[m][s * GP.n_ms_server + i].n_threads -= n

