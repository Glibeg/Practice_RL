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
        self.conv1 = nn.Conv2d(3, 16, 2)
        self.conv2 = nn.Conv2d(16, 8, 3)
        self.fc1 = nn.Linear(4 * 4 * 8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 2)
        self.conv2 = nn.Conv2d(16, 8, 3)
        self.encoderfc1 = nn.Linear(4 * 4 * 8, 2)
        self.encoderfc2 = nn.Linear(4 * 4 * 8, 2)
        self.decoderfc1 = nn.Linear(2, 128)
        self.convtrans1 = nn.ConvTranspose2d(8, 16, kernel_size = 3, stride= 2, padding = 1)
        self.convtrans2 = nn.ConvTranspose2d(16,3,kernel_size = 3, padding = 1)

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

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.sampling(mean, log_var)
        return mean, log_var, self.decode(z)

def vae_loss(inputs, outputs, mu, log_var):
    reconstruction_loss = F.mse_loss(inputs, outputs, reduction='mean')
    KL_loss = 1 + log_var - mu.pow(2) - log_var.exp()
    KL_loss = torch.mean(KL_loss)
    KL_loss *= -0.5
    return reconstruction_loss + KL_loss

env = gym.make('MiniGrid-Empty-8x8-v0')
env = gym_minigrid.wrappers.ImgObsWrapper(env)
action_size = env.action_space.n

dqn_value = critic(action_size).to(device)
dqn_optim = optim.Adam(params = dqn_value.parameters(), lr = 0.00025)

vae = VAE().to(device)
vae_optim = optim.Adam(params= vae.parameters(), lr = 0.00025)
agent = DQN.DQNagent_discrete(dqn_value, dqn_optim, action_size)

#tsne = TSNE(n_components=2, random_state=0)

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
    success_rate[1] += 1
    while not done:
        if agent.render:
            env.render()
        
        mean, log_var, recon_state = vae(state)
        encoded_states.append(mean.squeeze(0).cpu().detach().numpy())
        loss = vae_loss(state, recon_state, mean, log_var)
        vae_optim.zero_grad()
        loss.backward()
        vae_optim.step()

        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / EPS_DECAY)
        steps += 1
        action = agent.get_action(state, eps_threshold)
        next_state, reward, done, info = env.step(action.item())
        next_state = torch.tensor(next_state.transpose((2,0,1)), dtype=torch.float32).unsqueeze(0).to(device)
        if reward > 0 and done:
            success_rate[0] += 1
        score += reward

        agent.save_transition(state, action, done, reward, next_state)
        loss = agent.train_model()
        if loss != None:
            loss_list.append(loss)
        state = next_state

        if done:
            if (e+1)%20 == 0 or e < 5:
                encoded_states_ndarray = np.stack(encoded_states)
                #embeded_states = tsne.fit_transform(encoded_states_ndarray)
                plt.figure(figsize=(6,5))
                #plt.scatter(embeded_states[:,0], embeded_states[:,1], c=np.linspace(-5.0, 5.0, len(encoded_states)))
                plt.scatter(encoded_states_ndarray[:,0], encoded_states_ndarray[:,1], c=np.linspace(-5.0, 5.0, len(encoded_states)))
                plt.savefig("only_latent_VAE_episodes{0}.png".format(e+1)) 

            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            suc_rate = float(success_rate[0]) / success_rate[1] * 100
            print("episode : {:3d} | score_avg : {:3.2f} | success_rate : {:6.2f}% | loss = {:.3f}".format(e, score_avg, suc_rate, np.mean(loss_list)))

            if score_avg > 400:
                sys.exit()