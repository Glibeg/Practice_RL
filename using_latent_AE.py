from agent import DQN

import gym
import gym_minigrid
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
from sklearn.manifold import TSNE

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class critic(nn.Module):
    def __init__(self, actions):
        super(critic, self).__init__()
        self.critic_fc1 = nn.Linear(4, 24)
        self.critic_fc2 = nn.Linear(24, 24)
        self.critic_out = nn.Linear(24, actions)

    def forward(self, x):
        ch1 = torch.tanh(self.critic_fc1(x))
        ch2 = torch.tanh(self.critic_fc2(ch1))
        return self.critic_out(ch2)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 2)
        self.conv2 = nn.Conv2d(16, 8, 3)
        self.fc1 = nn.Linear(4 * 4 * 8, 4)
        self.fc2 = nn.Linear(4, 128)
        self.convtrans1 = nn.ConvTranspose2d(8, 16, kernel_size = 3, stride= 2, padding = 1)
        self.convtrans2 = nn.ConvTranspose2d(16,3,kernel_size = 3, padding = 1)

    def encode(self, x):
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        h = h.view(-1, 128)
        return self.fc1(h)

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        h = h.view(-1,8,4,4)
        h = torch.relu(self.convtrans1(h))
        return self.convtrans2(h)

    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z)

env = gym.make('MiniGrid-Empty-8x8-v0')
env = gym_minigrid.wrappers.ImgObsWrapper(env)
action_size = env.action_space.n

dqn_value = critic(action_size).to(device)
dqn_optim = optim.Adam(params = dqn_value.parameters(), lr = 0.00025)

AEncoder = AutoEncoder().to(device)
AE_optim = optim.Adam(params= AEncoder.parameters(), lr = 0.00025)
agent = DQN.DQNagent_discrete(dqn_value, dqn_optim, action_size)

tsne = TSNE(n_components=2, random_state=0)

scores, episodes = [], []
score_avg = 0
success_rate = [0, 0]
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
    state = torch.tensor(state.transpose((2,0,1)), dtype=torch.float32).unsqueeze(0).to(device)
    encoded_states = []

    z, recon_state = AEncoder(state)
    encoded_states.append(z.squeeze(0).cpu().detach().numpy())
    AE_loss = F.mse_loss(state, recon_state)
    AE_optim.zero_grad()
    AE_loss.backward()
    AE_optim.step()

    success_rate[1] += 1
    while not done:
        if agent.render:
            env.render()

        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / EPS_DECAY)
        steps += 1
        action = agent.get_action(z.detach(), eps_threshold)
        next_state, reward, done, info = env.step(action.item())
        next_state = torch.tensor(next_state.transpose((2,0,1)), dtype=torch.float32).unsqueeze(0).to(device)

        nz, recon_state = AEncoder(next_state)
        encoded_states.append(nz.squeeze(0).cpu().detach().numpy())
        AE_loss = F.mse_loss(next_state, recon_state)
        AE_optim.zero_grad()
        AE_loss.backward()
        AE_optim.step()

        if reward > 0 and done:
            success_rate[0] += 1
        score += reward

        agent.save_transition(z.detach(), action, done, reward, nz.detach())
        loss = agent.train_model()
        if loss != None:
            loss_list.append(loss)
        z = nz

        if done:
            if (e+1)%20 == 0 or e < 5:
                encoded_states_ndarray = np.stack(encoded_states)
                embeded_states = tsne.fit_transform(encoded_states_ndarray)
                plt.figure(figsize=(6,5))
                plt.scatter(embeded_states[:,0], embeded_states[:,1], c=np.linspace(-5.0, 5.0, len(encoded_states)))
                #plt.scatter(encoded_states_ndarray[:,0], encoded_states_ndarray[:,1], c=np.linspace(-5.0, 5.0, len(encoded_states)))
                plt.savefig("using_latent_AE_episodes{0}.png".format(e+1)) 

            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            suc_rate = float(success_rate[0]) / success_rate[1] * 100
            print("episode : {:3d} | score_avg : {:3.2f} | success_rate : {:6.2f}% | loss = {:.3f}".format(e, score_avg, suc_rate, np.mean(loss_list)))

            if score_avg > 400:
                sys.exit()