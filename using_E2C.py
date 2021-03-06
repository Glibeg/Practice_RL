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
import torch.distributions as distributions
import torchvision.transforms as T

import math
from sklearn.manifold import TSNE

from collections import namedtuple
from buffer.ReplayBuffer import ReplayMemory
import random

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
        self.critic_fc1 = nn.Linear(2, 24)
        self.critic_fc2 = nn.Linear(24, 24)
        self.critic_out = nn.Linear(24, actions)

    def forward(self, x):
        ch1 = torch.relu(self.critic_fc1(x))
        ch2 = torch.relu(self.critic_fc2(ch1))
        return self.critic_out(ch2)

class E2C(nn.Module):
    def __init__(self):
        super(E2C, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 2)
        self.conv2 = nn.Conv2d(16, 8, 3)
        self.encoderfc1 = nn.Linear(4 * 4 * 8, 2)
        self.encoderfc2 = nn.Linear(4 * 4 * 8, 2)
        self.decoderfc1 = nn.Linear(2, 128)
        self.convtrans1 = nn.ConvTranspose2d(8, 16, kernel_size = 3, stride= 2, padding = 1)
        self.convtrans2 = nn.ConvTranspose2d(16,3,kernel_size = 3, padding = 1)
        self.transitionfc1 = nn.Linear(2, 32)
        self.transitionfc2 = nn.Linear(32, 64)
        self.transitionout1 = nn.Linear(64, 2 * 2)
        self.transitionout2 = nn.Linear(64, 2 * 7)
        self.transitionout3 = nn.Linear(64, 2)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        h = h.view(-1, 128)
        return self.encoderfc1(h), self.encoderfc2(h)

    def decode(self, z):
        h = torch.relu(self.decoderfc1(z))
        h = h.view(-1,8,4,4)
        h = torch.relu(self.convtrans1(h))
        return self.convtrans2(h)

    def transition(self, z):
        h = torch.relu(self.transitionfc1(z))
        h = torch.relu(self.transitionfc2(h))
        A = self.transitionout1(h)
        B = self.transitionout2(h)
        o = self.transitionout3(h)
        return A.view(-1,2,2), B.view(-1,2,7), o

    def forward(self, x, u, nx):
        mean, log_var = self.encode(x)
        z = self.sampling(mean, log_var)
        rx = self.decode(z)
        A,B,o = self.transition(z)
        C = torch.bmm(torch.bmm(A, torch.diag_embed(torch.exp(0.5 * log_var))),A.transpose(1,2))

        nz_mean = torch.bmm(A , mean.unsqueeze(2)).squeeze(2) + torch.bmm(B ,u.unsqueeze(2)).squeeze(2) + o
        nzd = distributions.MultivariateNormal(nz_mean, C)
        nz = nzd.rsample()
        recon_nx = self.decode(nz)

        rnz_mean, rnz_log_var = self.encode(nx)
        rnzd = distributions.MultivariateNormal(rnz_mean, torch.diag_embed(torch.exp(0.5 * rnz_log_var)))

        return mean, log_var, rx, recon_nx, nzd, rnzd

def e2c_loss(mean, log_var, x, recon_x, nx, recon_nx, nzd, rnzd):
    alpha = 0.2
    lamb = 0.2
    reconstruction_loss1 = F.mse_loss(x, recon_x, reduction='mean')
    reconstruction_loss2 = F.mse_loss(nx, recon_nx, reduction='mean')
    KL_loss = 1 + log_var - mean.pow(2) - log_var.exp()
    KL_loss = torch.mean(KL_loss)
    KL_loss *= -0.5
    transition_loss = torch.mean(distributions.kl_divergence(nzd, rnzd))
    return reconstruction_loss1 + reconstruction_loss2 + alpha * KL_loss + lamb * transition_loss

env = gym.make('MiniGrid-Empty-8x8-v0')
env = gym_minigrid.wrappers.ImgObsWrapper(env)
action_size = env.action_space.n

#dqn_value = critic(action_size).to(device)
#dqn_optim = optim.Adam(params = dqn_value.parameters(), lr = 0.00025)

e2c = E2C().to(device)
e2c_optim = optim.Adam(params= e2c.parameters(), lr = 0.00025)
#agent = DQN.DQNagent_discrete(dqn_value, dqn_optim, action_size)

e2c_transition = namedtuple('e2c_transition',
                        ('state', 'action', 'next_state'))
e2c_memory = ReplayMemory(e2c_transition, 20000)

#tsne = TSNE(n_components=2, random_state=0)

scores, episodes = [], []
score_avg = 0
success_rate = [0, 0]
num_episodes = 1000
EPS_START = 1
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
    success_rate[1] += 1
    while not done:
        env.render()
        m, lv = e2c.encode(state)
        encoded_states.append(m.squeeze(0).detach().cpu().numpy())
        action =  torch.tensor([random.randrange(action_size)], device=device, dtype=torch.long)
        next_state, reward, done, info = env.step(action.item())
        next_state = torch.tensor(next_state.transpose((2,0,1)), dtype=torch.float32).unsqueeze(0).to(device)
        m, lv, rx, rnx, nzd, rnzd = e2c(state, F.one_hot(action, 7).to(torch.float32).to(device), next_state)
        e2closs = e2c_loss(m, lv, state, rx, next_state, rnx, nzd, rnzd)

        e2c_optim.zero_grad()
        e2closs.backward()
        e2c_optim.step()
        """
        e2c_memory.push(state, F.one_hot(action, 7).to(torch.float32).to(device), next_state)
        if len(e2c_memory) > 32:
            transitions = e2c_memory.sample(32)
            batch = e2c_transition(*zip(*transitions))

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            next_state_batch = torch.cat(batch.next_state)

            m, lv, rx, rnx, nz, rnz = e2c(state_batch, action_batch, next_state_batch)
            e2closs = e2c_loss(m, lv, state_batch, rx, next_state_batch, rnx, nz, rnz)

            e2c_optim.zero_grad()
            e2closs.backward()
            e2c_optim.step()"""

        if reward > 0 and done:
            success_rate[0] += 1
        score += reward

        state = next_state

        if done:
            if (e+1)%20 == 0 or e < 5:
                encoded_states_ndarray = np.stack(encoded_states)
                #embeded_states = tsne.fit_transform(encoded_states_ndarray)
                plt.figure(figsize=(6,5))
                #plt.scatter(embeded_states[:,0], embeded_states[:,1], c=np.linspace(-5.0, 5.0, len(encoded_states)))
                plt.scatter(encoded_states_ndarray[:,0], encoded_states_ndarray[:,1], c=np.linspace(-5.0, 5.0, len(encoded_states)))
                plt.savefig("using_E2C{0}.png".format(e+1)) 

            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            suc_rate = float(success_rate[0]) / success_rate[1] * 100
            print("episode : {:3d} | score_avg : {:3.2f} | success_rate : {:6.2f}% | loss = {:.3f}".format(e, score_avg, suc_rate, np.mean(loss_list)))

            if score_avg > 400:
                sys.exit()