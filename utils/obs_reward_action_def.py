import utils.global_parameters as GP

class OBSRWD:
    def __init__(self, ts, value):
        self.id = ts
        self.value = value
        self.major_reward = 0
        self.obs_delay = GP.obs_delay

class ACT:
    def __init__(self, ts, value, rate):
        self.id = ts
        self.value = value
        self.n_mapped_succ_rate = rate