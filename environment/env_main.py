import utils.global_parameters as GP
from utils.obs_reward_action_def import OBSRWD
from utils.logger import log



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
                    self.nfs[t].append(NF(t, i*GP.n_servers+j))

    def reset(self):
        pass
    def send_obs_reward(self, ts):
        obs = []
        for t in range(len(GP.c_r_ms)):
            for i in range(GP.n_servers):
                for j in range(GP.n_ms_server):
                    obs.append([self.nfs[t][i*GP.n_servers+j].lamda, self.nfs[t][i*GP.n_servers+j].n_threads])
        log.logger.debug('obs[%d]=\n%s' % (ts, str(obs)))
        return OBSRWD(ts, obs)

    def act(self, action):
        for act in action.value:
            [m, s, i, n] = act
            #log.logger.debug('[m,s,i,n]=[%d,%d,%d,%d]' % (m,s,i,n))
            self.nfs[m][s * GP.n_servers + i].lamda += 1
            self.nfs[m][s * GP.n_servers + i].n_threads += n