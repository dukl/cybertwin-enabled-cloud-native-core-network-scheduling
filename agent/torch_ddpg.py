import torch as T
import torch.nn.functional as F
import numpy as np
from agent.aiModels.network import ActorNetwork, CriticNetwork
from agent.buffer import ReplayBuffer
#from agent.replay_buffer import ReplayBuffer
from utils.actions_definition import ACTIONS
import utils.auto_scaling_settings as ACS
from agent.aiModels.utils import create_directory, plot_learning_curve, scale_action


device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class DDPG:
    def __init__(self, alpha, beta, state_dim, action_dim, actor_fc1_dim,
                 actor_fc2_dim, critic_fc1_dim, critic_fc2_dim, ckpt_dir,
                 gamma=0.99, tau=0.005, action_noise=0.1, max_size=1000000,
                 batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.checkpoint_dir = ckpt_dir

        self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.target_actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.critic = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                    fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_critic = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                           fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim,
                                   batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

        self.last_obs_id = -1
        self.pending_s = None
        self.pending_a = None

    def reset(self):
        self.last_obs_id = -1
        self.pending_s = None
        self.pending_a = None

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic_params, target_critic_params in zip(self.critic.parameters(),
                                                       self.target_critic.parameters()):
            target_critic_params.data.copy_(tau * critic_params + (1 - tau) * target_critic_params)

    #def remember(self, state, action, reward, state_, done):
    #    self.memory.store_transition(state, action, reward, state_, done)

    def remember(self, s, a, r):
        if self.pending_s is not None:
            self.memory.store_transition(self.pending_s, self.pending_a, r, s, False)
        self.pending_s, self.pending_a = s, a

    def choose_action(self, observation, train=True):
        #print(observation.shape)
        observation = observation.reshape((1, observation.shape[0]))
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(device)
        action = self.actor.forward(state).squeeze()

        if train:
            noise = T.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                             dtype=T.float).to(device)
            action = T.clamp(action + noise, -1, 1)
        self.actor.train()

        return action.detach().cpu().numpy()

    def learn(self):
        if not self.memory.ready():
            return
        print('Training ...')

        states, actions, reward, states_, terminals = self.memory.sample_buffer()
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.float).to(device)
        rewards_tensor = T.tensor(reward, dtype=T.float).to(device)
        next_states_tensor = T.tensor(states_, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            q_ = self.target_critic.forward(next_states_tensor, next_actions_tensor).view(-1)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_
        q = self.critic.forward(states_tensor, actions_tensor).view(-1)

        critic_loss = F.mse_loss(q, target.detach())
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        new_actions_tensor = self.actor.forward(states_tensor)
        actor_loss = -T.mean(self.critic(states_tensor, new_actions_tensor))
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def save_models(self, episode):
        self.actor.save_checkpoint(self.checkpoint_dir + 'Actor/DDPG_actor_{}.pth'.format(episode))
        print('Saving actor network successfully!')
        self.target_actor.save_checkpoint(self.checkpoint_dir +
                                          'Target_actor/DDPG_target_actor_{}.pth'.format(episode))
        print('Saving target_actor network successfully!')
        self.critic.save_checkpoint(self.checkpoint_dir + 'Critic/DDPG_critic_{}'.format(episode))
        print('Saving critic network successfully!')
        self.target_critic.save_checkpoint(self.checkpoint_dir +
                                           'Target_critic/DDPG_target_critic_{}'.format(episode))
        print('Saving target critic network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Actor/DDPG_actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(self.checkpoint_dir +
                                          'Target_actor/DDPG_target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')
        self.critic.load_checkpoint(self.checkpoint_dir + 'Critic/DDPG_critic_{}'.format(episode))
        print('Loading critic network successfully!')
        self.target_critic.load_checkpoint(self.checkpoint_dir +
                                           'Target_critic/DDPG_target_critic_{}'.format(episode))
        print('Loading target critic network successfully!')

    def choose_action_with_delayed_obs(self, obs_on_road, ts):
        avai_obs = None
        arrived_obs = []
        for i, obs in enumerate(obs_on_road):
            if obs[0] + obs[1] <= ts:
                arrived_obs.append(obs)
        max_delay = 0
        index, is_avai_obs = 0, False
        for i, obs in enumerate(arrived_obs):
            if obs[0] > self.last_obs_id:
                if obs[0] + obs[1] >= max_delay:
                    max_delay = obs[0] + obs[1]
                    index = i
                    is_avai_obs = True
                    self.last_obs_id = obs[0]
        if is_avai_obs is True:
            avai_obs = arrived_obs[index]
        if avai_obs == None:
            return None  # No action

        for obs in arrived_obs:
            obs_on_road.remove(obs)

        #print(avai_obs[2].shape)
        act_value = self.choose_action(avai_obs[2], train=True)
        self.remember(avai_obs[2], act_value, avai_obs[3])
        self.learn()

        act_value_ = scale_action(act_value.copy(), 1, -1)
        #print(act_value_)
        #print(act_value_.shape)

        v_s_out = act_value_.reshape((ACS.n_node * (len(ACS.t_NFs) - 1), ACS.n_max_inst))

        ret_a = ACTIONS()
        ret_a.v_s = v_s_out
        ret_a.h_s = np.zeros((ACS.n_node, len(ACS.t_NFs) - 1)).astype('int32')

        return ret_a