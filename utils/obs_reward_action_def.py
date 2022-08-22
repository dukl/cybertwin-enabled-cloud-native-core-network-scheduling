import utils.global_parameters as GP

class OBSRWD:
    def __init__(self, ts, value, reward):
        self.id = ts
        self.value = value
        self.major_reward = reward
        self.obs_delay = GP.obs_delay

class ACT:
    def __init__(self, ts, value, rate, total_reqs):
        self.id = ts
        self.value = value
        self.n_mapped_succ_rate = rate
        self.total_reqs = total_reqs