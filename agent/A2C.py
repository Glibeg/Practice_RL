import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class A2C_discrete_agent():
    def __init__(self, actor, critic, AC_optim, action_size):
        """actor : softmax output, critic : state value estimator"""
        self.render = True

        self.action_size = action_size
        self.discount_factor = 0.99

        self.actor = actor
        self.critic = critic
        self.optimizer = AC_optim

    def get_action(self, state):
        policy = self.actor(state)
        policy = policy[0].cpu().detach().numpy()
        return np.random.choice(self.action_size, 1, p = policy)[0]
        
    def train_model(self, state, action, reward, next_state, done):
        policy = self.actor(state)
        value = self.critic(state)
        next_value = self.critic(next_state)
        td_target = reward + (1-done) * self.discount_factor * next_value[0]
        
        one_hot_action = F.one_hot(torch.tensor([action]).to(device).long(), self.action_size)
        action_prob = torch.sum((one_hot_action * policy), 1)
        advantage = (td_target - value[0]).detach()
        actor_loss = torch.log(action_prob + 1e-5) * advantage
        actor_loss = -torch.mean(actor_loss)

        critic_loss = 0.5 * torch.square(td_target.detach() - value[0])
        critic_loss = torch.mean(critic_loss)

        loss = 0.2 * actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.)
        self.optimizer.step()
        return loss.cpu().detach().numpy()