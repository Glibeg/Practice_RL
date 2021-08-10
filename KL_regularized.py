import gym
import os
import sys
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import dm_control.suite as suite
import dm_control.viewer as viewer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions.normal import Normal
import torch.distributions

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#torch.set_default_dtype(torch.float32)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class PolicyNet(nn.Module):
    def __init__(self, observations, actions):
        super(PolicyNet, self).__init__()
        self.policy_fc1 = nn.Linear(observations, 128)
        self.policy_fc2 = nn.Linear(128, 256)
        self.policy_fc3 = nn.Linear(256, 256)
        self.policy_mean_fc1 = nn.Linear(256, 128)
        self.policy_mean_out = nn.Linear(128, actions)
        self.policy_logstd_fc1 = nn.Linear(256, 128)
        self.policy_logstd_out = nn.Linear(128, actions)

    def forward(self, x):#, deterministic = False, with_logprob = True):
        h = torch.relu(self.policy_fc1(x))
        h = torch.relu(self.policy_fc2(h))
        h = torch.relu(self.policy_fc3(h))
        mean = torch.tanh(self.policy_mean_out(torch.relu(self.policy_mean_fc1(h))))
        std = torch.exp(torch.clamp(self.policy_logstd_out(torch.relu(self.policy_logstd_fc1(h))), LOG_STD_MIN, LOG_STD_MAX))
        return mean, std
        """
        pi_distribution = Normal(mean, std)
        if deterministic:
            pi_action = mean
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis = -1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        return pi_action, logp_pi
        """

class ValueNet(nn.Module):
    def __init__(self, observations, actions):
        super(ValueNet, self).__init__()
        self.critic_fc1 = nn.Linear(observations + actions, 128)
        self.critic_fc2 = nn.Linear(128, 256)
        self.critic_fc3 = nn.Linear(256, 256)
        self.critic_fc4 = nn.Linear(256, 128)
        self.critic_out = nn.Linear(128, 1)

    def forward(self, x, a):
        ch1 = torch.relu(self.critic_fc1(torch.cat([x,a], dim = 1)))
        ch2 = torch.relu(self.critic_fc2(ch1))
        ch3 = torch.relu(self.critic_fc3(ch2))
        ch4 = torch.relu(self.critic_fc4(ch3))
        return self.critic_out(ch4)

class KL():
    def __init__(self, states_dim, default_states_dim, action_space):
        self.learning_rate = 1e-3
        self.memory_capacity = 10000
        self.batch_size = 64
        self.action_space = action_space
        self.alpha = 1
        self.gamma = 0.9
        self.polyak = 0.995
        self.update_every = 1
        self.update_after = 1000
        self.render = True

        self.policy_net = PolicyNet(states_dim,action_space.shape[0]).to(device)
        self.policy_optim = optim.Adam(self.policy_net.parameters(), lr= self.learning_rate)

        self.default_policy_net = PolicyNet(default_states_dim, action_space.shape[0]).to(device)
        self.default_policy_optim = optim.Adam(self.default_policy_net.parameters(), lr = self.learning_rate)

        self.value_net = ValueNet(states_dim,action_space.shape[0]).to(device)
        self.value_net_target = ValueNet(states_dim,action_space.shape[0]).to(device)
        self.value_net_target.load_state_dict(self.value_net.state_dict())
        self.value_optim = optim.Adam(self.value_net.parameters(), lr= self.learning_rate)

        for param in self.value_net_target.parameters():
            param.requires_grad = False
        self.replay_buffer = ReplayMemory(self.memory_capacity)

    def get_action(self, state, deterministic = False):
        with torch.no_grad():
            mean, std = self.policy_net(state)
            eps = 0 if deterministic else torch.randn_like(std)
            return torch.max(torch.min(mean + std * eps, torch.tensor(self.action_space.maximum, dtype = torch.float32).to(device)), torch.tensor(self.action_space.minimum, dtype = torch.float32).to(device))

    def train_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        qloss_list, ploss_list = [],[]
        for c in range(self.update_every):
            transitions = self.replay_buffer.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            next_state_batch = torch.cat(batch.next_state)
            done_batch = torch.cat(batch.done)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            mean, std = self.policy_net(state_batch)
            default_mean, default_std = self.default_policy_net(state_batch)
            pi_distribution = Normal(mean, std)
            pi_zero_distribution = Normal(default_mean, default_std)
            
            next_mean, next_std = self.policy_net(next_state_batch)
            next_mean.detach_()
            next_std.detach_()
            next_default_mean, next_default_std = self.default_policy_net(next_state_batch)
            next_pi_distribution = Normal(next_mean, next_std)
            next_pi_zero_distribution = Normal(next_default_mean, next_default_std)

            kl_div_term = torch.distributions.kl.kl_divergence(pi_distribution, pi_zero_distribution).sum(axis = -1, keepdim = True)
            next_kl_div_term = torch.distributions.kl.kl_divergence(next_pi_distribution, next_pi_zero_distribution).sum(axis = -1, keepdim = True)

            with torch.no_grad():
                target_action = next_pi_distribution.sample()
                target_action = torch.max(torch.min(target_action, torch.tensor(self.action_space.maximum, dtype=torch.float32).to(device)), torch.tensor(self.action_space.minimum, dtype=torch.float32).to(device))
                value_target = self.value_net_target(next_state_batch, target_action)
                td_target = reward_batch - self.alpha * kl_div_term.detach() + self.gamma * (1 - done_batch) * (value_target - self.alpha * next_kl_div_term.detach())
            
            self.value_optim.zero_grad()
            state_action = self.value_net(state_batch, action_batch)
            q_loss = F.mse_loss(state_action, td_target)
            q_loss.backward()
            self.value_optim.step()
            qloss_list.append(q_loss.cpu().detach().numpy())

            self.policy_optim.zero_grad()
            self.default_policy_optim.zero_grad()
            value = self.value_net(state_batch, action_batch)
            policy_loss = (self.alpha * (kl_div_term + next_kl_div_term) - value).mean()
            policy_loss.backward()
            self.policy_optim.step()
            self.default_policy_optim.step()
            ploss_list.append(policy_loss.cpu().detach().numpy())

            with torch.no_grad():
                for param, target_param in zip(self.value_net.parameters(), self.value_net_target.parameters()):
                    target_param.data.mul_(self.polyak)
                    target_param.data.add_(param.data * (1-self.polyak))

        return  np.mean(qloss_list), np.mean(ploss_list)

def state_preprocessing(state):
    return torch.tensor(state.copy(), dtype = torch.float32).unsqueeze(0).to(device)

#env = gym.make('FetchReach-v1')
env = suite.load(domain_name = 'quadruped', task_name = 'walk', environment_kwargs = {'flat_observation' : True})
"""time_step = env.reset()
print(state_preprocessing(time_step.observation['observations']))"""
#env.env.reward_type = 'dense'
#env = gym.wrappers.filter_observation.FilterObservation(env, filter_keys=['observation', 'desired_goal'])
#env = gym.wrappers.flatten_observation.FlattenObservation(env)
agent = KL(78, 78, env.action_spec())
#agent.policy_net.load_state_dict(torch.load('./pnet.pth'))
#agent.value_net.load_state_dict(torch.load('./vnet.pth'))


scores, episodes = [], []
score_avg = 0

def policy_by_agent(time_step):
    torch_action =  agent.get_action(state_preprocessing(time_step.observation['observations']))
    return torch_action.squeeze(0).cpu().detach().numpy()

#viewer.launch(env, policy=policy_by_agent)
#exit(0)
initial_count = 0
num_episodes = 1000
for e in range(num_episodes):
    done = False
    score = 0
    qloss_list, ploss_list = [], []
    time_step = env.reset()
    state = state_preprocessing(time_step.observation['observations'])

    for c in count():
        if initial_count <= agent.update_after:
            action = np.random.uniform(env.action_spec().minimum, env.action_spec().maximum, env.action_spec().shape)
            action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
            initial_count += 1
        else:
            action = agent.get_action(state)
        
        time_step = env.step(action.squeeze(0).cpu().detach().numpy())
        #print(time_step)
        next_state = state_preprocessing(time_step.observation['observations'])
        reward = time_step.reward
        done = time_step.discount if time_step.step_type == 1 else 0.0
        score += reward
        reward = torch.tensor([[reward]], dtype=torch.float32).to(device)
        done = torch.tensor([[done]]).to(device)
        agent.replay_buffer.push(state, action, next_state, reward, done)
        state = next_state
        
        if c % agent.update_every == 0:
            qloss, ploss = agent.train_model()
            if qloss != None:
                qloss_list.append(qloss)
            if ploss != None:
                ploss_list.append(ploss)

        if time_step.step_type == 2:
            #score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode : {:3d} | end at : {:3d} steps | score : {:3.3f} | q_loss = {:3.3f} | p_loss = {:.3f} ".format(e, c, score, np.mean(qloss_list), np.mean(ploss_list)))
            #if score_avg > 900:
            #    sys.exit()
            break
    torch.save(agent.policy_net.state_dict(), './pnet.pth')
    torch.save(agent.value_net.state_dict(), './vnet.pth')
env.close()