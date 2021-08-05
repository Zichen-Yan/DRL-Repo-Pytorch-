from collections import namedtuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'is_terminals'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, num_action, num_state, hidden_layer=64):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(num_state, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc_mu = nn.Linear(hidden_layer, num_action)
        self.fc_std = nn.Linear(hidden_layer, num_action)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.tanh(self.fc_mu(x))
        std = self.softplus(self.fc_std(x)) + 1e-3
        return mu, std

class Critic(nn.Module):
    def __init__(self, num_state, hidden_layer=64):
        super(Critic, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(num_state, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class CPPO(object):
    gamma = 0.99
    clip_param = 0.2
    max_grad_norm = 0.5

    def __init__(self, env, a_lr=1e-3, c_lr=2e-3):
        super(CPPO, self).__init__()
        self.actor_net = Actor(num_state=env.observation_space.shape[0], num_action=env.action_space.shape[0])
        self.critic_net = Critic(num_state=env.observation_space.shape[0])
        self.buffer = []
        self.vloss = []
        self.aloss = []
        self.training_step = 0

        self.n_a=env.action_space.shape[0]
        self.n_s=env.observation_space.shape[0]

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=a_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=c_lr)

        self.root = os.path.dirname(os.path.realpath(__file__))
        self.save_dir = os.path.join(self.root, "CPPO_net_param/")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def action(self, state):
        state = torch.FloatTensor(state).view(1, -1)
        mu, std = self.actor_net(state)
        dist = Normal(mu, std)
        action = dist.sample()

        return torch.squeeze(action).tolist(), dist.log_prob(action)

    def evaluate(self, state, action):
        mu, std = self.actor_net(state)  # size(action_means)=([batch, action_dim])
        dist = Normal(mu, std)
        action_log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        value = self.critic_net(state)

        return value, action_log_prob, dist_entropy

    def save_param(self):
        torch.save(self.actor_net.state_dict(), self.save_dir + '/actor_net_params.pkl')
        torch.save(self.critic_net.state_dict(), self.save_dir + '/critic_net_params.pkl')

    def load_params(self):
        self.actor_net.load_state_dict(torch.load(self.save_dir + '/actor_net_params.pkl'))
        self.critic_net.load_state_dict(torch.load(self.save_dir + '/critic_net_params.pkl'))

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float32)
        next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float32)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.float32).view(-1, self.n_a).detach()
        reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float32).view(-1, 1)
        old_action_log_prob = [t.a_log_prob for t in self.buffer]
        old_action_log_prob = torch.stack(old_action_log_prob).view(-1, self.n_a).detach()

        for i in range(10):
            # compute advantage
            with torch.no_grad():
                value_target = reward + self.gamma * self.critic_net(next_state)
                advantage = value_target - self.critic_net(state)

            _, new_action_log_prob, dist_entropy = self.evaluate(state, action)
            # Surrogate Loss
            ratio = torch.exp(new_action_log_prob - old_action_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

            aloss = -torch.min(surr1, surr2) - 0.01*dist_entropy
            self.actor_optimizer.zero_grad()
            aloss.mean().backward()
            nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            vloss = F.mse_loss(value_target, self.critic_net(state))
            self.critic_optimizer.zero_grad()
            vloss.backward()
            nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            self.vloss.append(vloss.detach().numpy().tolist())
            self.aloss.append(aloss.mean().detach().numpy().tolist())

            self.training_step += 1

        del self.buffer[:]  # clear experience


if __name__ == '__main__':
    agent = CPPO(2, 2)
    pass
