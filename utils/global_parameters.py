import numpy.random

n_episode = 1000
n_time_steps = 100
agent_type = 'ddqnMlp' # ['nddqn', 'ddqnMlp']
obs_delay = 0

n_servers = 4
n_ms_server = 3
ypi_max = 8

c_r_ms = [25, 36, 34, 24, 25, 15]
psi_ms = [92, 52, 99, 68, 78, 112]
w_ms   = [10, 12, 9, 5, 6, 8]
lamda_ms = [100, 104, 110, 210, 120, 130]

n_cpu_core = 3
cpu = 8000

msc = [
    [1,2],
    [1,2,5],
    [0,2,4,3,1],
    [0,4,1,5],
    [0,1,3,2,5],
    [3,2,0],
    [4,2,3,0],
    [2,3,4,5]
]

arrive_rate = numpy.random.normal(5, 1, 10)

one_step_time = 1500

beta_r = 2