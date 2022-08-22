import random
import numpy
from utils.logger import log

import utils.global_parameters as GP

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
        reqs = self.receive_requests()
        obs_env = [b for a in obs[0].value for b in a]
        obs_req = []
        for i in range(len(GP.msc)):
            for j in range(reqs[i]):
                for ms in GP.msc[i]:
                    obs_req += [i, reqs[i], ms]
                    obs_input = numpy.array(obs_env + obs_req)
                    log.logger.debug('obs_input=\n%s' % (str(obs_input)))


