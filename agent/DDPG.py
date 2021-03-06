from buffer.ReplayBuffer import ReplayMemory
import copy
import torch
import torch.nn.functional as F
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class DDPG():
    def __init__(self, policy_net, value_net, policy_optim, value_optim, action_space):
        """policy_net : continuous actor / value_net : state-action value estimator"""
        self.memory_capacity = 10000
        self.batch_size = 32
        self.gamma = 0.9
        self.polyak = 0.995
        self.render = True
        self.action_space = action_space

        self.policy_net = policy_net
        self.policy_net_target = copy.deepcopy(self.policy_net)
        self.policy_optim = policy_optim

        self.value_net = value_net
        self.value_net_target = copy.deepcopy(self.value_net)
        self.value_optim = value_optim

        for param in self.policy_net_target.parameters():
            param.requires_grad = False
        for param in self.value_net_target.parameters():
            param.requires_grad = False

        self.replay_buffer = ReplayMemory(Transition, self.memory_capacity)

    def get_action(self, state, noise = 0):
        with torch.no_grad():
            action = self.policy_net(state)
            noise_v = torch.randn(self.action_space.shape).to(device) * noise
            return torch.max(torch.min(action + noise_v, torch.tensor(self.action_space.high).to(device)), torch.tensor(self.action_space.low).to(device))

    def save_transition(self, state, action, done, reward, next_state):
        reward = torch.tensor([[reward]]).to(device)
        done = torch.tensor([[1 if done else 0]]).to(device)
        self.replay_buffer.push(state, action, next_state, reward, done)

    def train_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)
        next_state_batch = torch.cat(batch.next_state)

        with torch.no_grad():
            td_target = reward_batch + self.gamma * (1 - done_batch) * self.value_net_target(next_state_batch, self.policy_net_target(next_state_batch))
        
        self.value_optim.zero_grad()
        state_action = self.value_net(state_batch, action_batch)
        q_loss = F.mse_loss(state_action, td_target)
        q_loss.backward()
        self.value_optim.step()
        
        self.policy_optim.zero_grad()
        p_loss = self.value_net(state_batch, self.policy_net(state_batch))
        p_loss = -p_loss.mean()
        p_loss.backward()
        self.policy_optim.step()

        with torch.no_grad():
            for param, target_param in zip(self.policy_net.parameters(), self.policy_net_target.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_(param.data * (1-self.polyak))
            for param, target_param in zip(self.value_net.parameters(), self.value_net_target.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_(param.data * (1-self.polyak))
        
        return q_loss.cpu().detach().numpy(), p_loss.cpu().detach().numpy()