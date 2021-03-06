from agent import DQN

import gym
import sys
import numpy as np
import matplotlib
import itertools
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import math

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class critic(nn.Module):
    def __init__(self, observations, actions):
        super(critic, self).__init__()
        self.critic_fc1 = nn.Linear(observations, 24)
        self.critic_fc2 = nn.Linear(24, 24)
        self.critic_out = nn.Linear(24, actions)

    def forward(self, x):
        ch1 = torch.tanh(self.critic_fc1(x))
        ch2 = torch.tanh(self.critic_fc2(ch1))
        return self.critic_out(ch2)

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

dqn_value = critic(state_size, action_size).to(device)
dqn_optim = optim.Adam(params = dqn_value.parameters(), lr = 0.001)

agent = DQN.DQNagent_discrete(dqn_value, dqn_optim, action_size)

scores, episodes = [], []
score_avg = 0

num_episodes = 1000
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
steps = 0
for e in range(num_episodes):
    done = False
    score = 0
    loss_list = []
    state = env.reset()
    state = torch.tensor(np.reshape(state, [1,state_size]),dtype=torch.float32).to(device)

    while not done:
        if agent.render:
            env.render()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / EPS_DECAY)
        steps += 1
        action = agent.get_action(state, eps_threshold)
        next_state, reward, done, info = env.step(action.item())
        next_state = torch.tensor(np.reshape(next_state, [1, state_size]),dtype=torch.float32).to(device)

        score += reward
        reward = 0.1 if not done or score == 500 else -1.

        agent.save_transition(state, action, done, reward, next_state)
        loss = agent.train_model()
        if loss != None:
            loss_list.append(loss)
        state = next_state

        if done:
            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode : {:3d} | score_avg : {:3.2f} | loss = {:.3f}".format(e, score_avg, np.mean(loss_list)))

            if score_avg > 400:
                sys.exit()