from agent import A2C

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

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class actor(nn.Module):
    def __init__(self, observations, actions):
        super(actor, self).__init__()
        self.actor_fc1 = nn.Linear(observations, 24)
        self.actor_out = nn.Linear(24, actions)

    def forward(self, x):
        ah = torch.tanh(self.actor_fc1(x))
        return torch.softmax(self.actor_out(ah),dim=1)

class critic(nn.Module):
    def __init__(self, observations):
        super(critic, self).__init__()
        self.critic_fc1 = nn.Linear(observations, 24)
        self.critic_fc2 = nn.Linear(24, 24)
        self.critic_out = nn.Linear(24, 1)

    def forward(self, x):
        ch1 = torch.tanh(self.critic_fc1(x))
        ch2 = torch.tanh(self.critic_fc2(ch1))
        return self.critic_out(ch2)

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

a2c_actor = actor(state_size, action_size).to(device)
a2c_critic = critic(state_size).to(device)
a2c_params = itertools.chain(a2c_actor.parameters(), a2c_critic.parameters())
a2c_optim = optim.Adam(params = a2c_params, lr = 0.001)

agent = A2C.A2C_discrete_agent(a2c_actor, a2c_critic, a2c_optim, action_size)

scores, episodes = [], []
score_avg = 0

num_episodes = 1000
for e in range(num_episodes):
    done = False
    score = 0
    loss_list = []
    state = env.reset()
    state = torch.tensor(np.reshape(state, [1,state_size]),dtype=torch.float32).to(device)

    while not done:
        if agent.render:
            env.render()
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = torch.tensor(np.reshape(next_state, [1, state_size]),dtype=torch.float32).to(device)

        score += reward
        reward = 0.1 if not done or score == 500 else -1.

        loss = agent.train_model(state, action, reward, next_state, done)
        loss_list.append(loss)
        state = next_state

        if done:
            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode : {:3d} | score_avg : {:3.2f} | loss = {:.3f}".format(e, score_avg, np.mean(loss_list)))

            if score_avg > 400:
                sys.exit()