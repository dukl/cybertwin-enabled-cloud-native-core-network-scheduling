import numpy.random

n_episode = 1
n_time_steps = 2
agent_type = 'nddqn' # no-delay-dqn
obs_delay = 0

n_servers = 10
n_ms_server = 10

c_r_ms = [25, 36, 34, 24, 25, 15]
psi_ms = [92, 52, 99, 68, 78, 112]

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

arrive_rate = numpy.random.normal(4, 0, 10)