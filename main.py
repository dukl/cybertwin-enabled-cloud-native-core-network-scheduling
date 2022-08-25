import random

from utils.logger import log
from environment.env_main import ENV
import utils.global_parameters as GP
import utils.system_inner as SI
from agent.no_delay_dqn import NDDQN
from agent.delay_dqn_mlp import DDQNMLP
from agent.turn_based_agent import TBA
import results.running_value as RV

if __name__ == '__main__':
    #log.logger.debug('[Line-1][Initialize env and agent]')
    log.logger.debug('Experimental Parameters: n_episode=%d, n_time_steps=%d, agent_type=%s' % (GP.n_episode, GP.n_time_steps, GP.agent_type))
    env, agent = ENV(), None
    if GP.agent_type is 'nddqn':
        agent = NDDQN()
        GP.obs_delay = 0
    if GP.agent_type is 'ddqnMlp':
        GP.obs_delay = 1
        agent = DDQNMLP(GP.obs_delay)
    if GP.agent_type is 'turnAgt':
        GP.obs_delay = 1 #random.randint(1,2)
        agent = TBA()
    for ep in range(GP.n_episode):
        SI.RESET(env, agent)
        RV.episode_reward.append(0)
        for ts in range(GP.n_time_steps):
            RV.mapped_succ_rate.append(0)
            log.logger.debug('[line-9][Training Episode - %d][Time Step - %d]' % (ep, ts))
            obs_rwd = env.send_obs_reward(ts)
            RV.obs_on_road.append(obs_rwd)
            action = agent.receive_observation_s(SI.CHECK_OBSERVATIONS(ts), ts)
            env.act(action)
        log.logger.debug('Episode-%d reward: %f' % (ep, RV.episode_reward[-1]))




