import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, num_state, num_action, action_bound=1.0, n_latent_var=128):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(num_state, n_latent_var)
        self.fc2 = nn.Linear(n_latent_var, n_latent_var)

        self.mu = nn.Linear(n_latent_var, num_action)
        self.std = nn.Linear(n_latent_var, num_action)
        self.v = nn.Linear(n_latent_var, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()
        self.action_bound = action_bound

    def actor(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.action_bound*self.tanh(self.mu(x))
        std = self.softplus(self.std(x)) + 1e-3
        return mu, std

    def critic(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        v = self.v(x)
        return v

    def forward(self, x):
        mu, std = self.actor(x)
        v = self.critic(x)

        return mu, std, v

    def act(self, state):
        state = torch.from_numpy(state).float().view(1, -1)
        mu, std = self.actor(state)
        dist = Normal(mu, std)
        action = dist.sample()

        return action.detach().numpy().flatten(), dist.log_prob(action).detach().numpy().flatten()

    def set_weights(self, params):
        self.load_state_dict(params)

    def get_weights(self):
        return self.state_dict()
