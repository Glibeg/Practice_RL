import gym
import sys
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent.DDPG import DDPG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(nn.Module):
    def __init__(self, h, w, outputs):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return torch.tanh(self.head(x.reshape(x.size(0), -1)))

class ValueNet(nn.Module):
    def __init__(self, h, w, actions):
        super(ValueNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32 + actions
        self.fc1 = nn.Linear(linear_input_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(torch.cat((x.reshape(x.size(0), -1), a), dim=1)))
        return self.fc2(x)

env = gym.make('CarRacing-v0')
state_h, state_w, _ = env.observation_space.shape
action_size = env.action_space.shape[0]
policyNet = PolicyNet(state_h, state_w, action_size).to(device)
policyOptim = optim.Adam(policyNet.parameters(), lr= 0.00025)
valueNet = ValueNet(state_h, state_w, action_size).to(device)
valueOptim = optim.Adam(valueNet.parameters(), lr= 0.00025)
agent = DDPG(policyNet, valueNet, policyOptim, valueOptim, env.action_space)

scores, episodes = [], []
score_avg = 0

num_episodes = 1000
for e in range(num_episodes):
    done = False
    score = 0
    qloss_list, ploss_list = [], []
    statet = env.reset()
    statet = torch.tensor(statet.copy().transpose((2,0,1)), dtype= torch.float32).unsqueeze(0).to(device)
    statec , _, _, _ = env.step(env.action_space.sample())
    statec = torch.tensor(statec.copy().transpose((2,0,1)), dtype= torch.float32).unsqueeze(0).to(device)
    state = statec - statet
    for c in count():
        if agent.render:
            env.render()
        action = agent.get_action(state, 0.1)
        statet, reward, done, info = env.step(action.squeeze(0).cpu().detach().numpy())
        score += reward
        statet = torch.tensor(statet.copy().transpose((2,0,1)), dtype= torch.float32).unsqueeze(0).to(device)
        next_state = statet - statec

        agent.save_transition(state, action, done, reward, next_state)
        state = next_state
        statec = statet
        
        qloss, ploss = agent.train_model()
        if qloss != None:
            qloss_list.append(qloss)
        if ploss != None:
            ploss_list.append(ploss)

        if done:
            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode : {:3d} | score_avg : {:3.2f} | q_loss = {:3.3f} | p_loss = {:.3f}".format(e, score_avg, np.mean(qloss_list), np.mean(ploss_list)))
            if score_avg > 900:
                sys.exit()
            break