import numpy.random

n_episode = 1000
n_time_steps = 100
agent_type = 'nddqn' # ['nddqn', 'ddqnMlp', 'turnAgt']
obs_delay = 0

n_servers = 2
n_ms_server = 2
ypi_max = 2

c_r_ms = [25, 36, 34, 24, 25, 15]
psi_ms = [92, 52, 99, 68, 78, 112]
w_ms   = [10, 10, 10, 10, 10, 10]
lamda_ms = [20, 20, 20, 20, 20, 20]

n_cpu_core = 3
cpu = 8000

msc = [
    [4,5],
    #[1,2,5],
    [0,2,4,3,1],
    #[0,4,1,5],
    [0,1,3,2,5]
    #[3,2,0],
    #[4,2,3,0],
    #[2,3,4,5]
]

n_reqs_per_msc = 10

arrive_rate = numpy.random.normal(n_reqs_per_msc, 0, 10)

one_step_time = 40

beta_r = 10

n_ensemble = 5