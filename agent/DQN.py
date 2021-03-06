from buffer.ReplayBuffer import ReplayMemory
import random
import torch
import torch.nn.functional as F
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class DQNagent_discrete():
    def __init__(self, value_net, optimizer, action_size):
        """value_net : state-action value estimator"""
        self.memory_capacity = 10000
        self.batch_size = 128
        self.gamma = 0.9
        self.render = True
        self.action_size = action_size

        self.value_net = value_net
        self.optimizer = optimizer

        self.replay_buffer = ReplayMemory(Transition, self.memory_capacity)

    def get_action(self, state, eps):
        with torch.no_grad():
            if random.random() < eps:
                return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
            else:
                return self.value_net(state).max(1)[1].view(1, 1)

    def save_transition(self, state, action, done, reward, next_state):
        reward = torch.tensor([[reward]]).to(device)
        done = torch.tensor([[1 if done else 0]]).to(device)
        self.replay_buffer.push(state, action, next_state, reward, done)

    def train_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        state_action_values = self.value_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_state_values = self.value_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
            td_target = reward_batch + self.gamma * next_state_values * (1 - done_batch)
        loss = F.smooth_l1_loss(state_action_values, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.cpu().detach().numpy()