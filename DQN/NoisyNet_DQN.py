import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import random
import os
import math
from copy import deepcopy
from utils import ReplayMemory


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,  # w
            self.bias_mu + self.bias_sigma * self.bias_epsilon,  # b
        )  # y=wx+b

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


class Net(nn.Module):
    def __init__(self, N_STATES: int, N_ACTIONS: int):
        """Initialization."""
        super(Net, self).__init__()

        self.feature = nn.Linear(N_STATES, 64)
        self.noisy_layer1 = NoisyLinear(64, 64)
        self.noisy_layer2 = NoisyLinear(64, N_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = F.relu(self.feature(x))
        hidden = F.relu(self.noisy_layer1(feature))
        out = self.noisy_layer2(hidden)

        return out

    def reset_noise(self):
        """Reset all noisy layers."""
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()


class DQN(object):
    GAMMA = 0.9  # reward discount
    TARGET_REPLACE_ITER = 100  # target update frequency

    def __init__(self, num_state, num_action, batch_size=32, lr=1e-3, memory_size=2000, Transition=None):
        super(DQN, self).__init__()
        self.n_a = num_action
        self.n_s = num_state
        self.batch_size = batch_size
        self.Transition = Transition

        self.eval_net = Net(N_STATES=num_state, N_ACTIONS=num_action)
        self.target_net = self.target_net = deepcopy(self.eval_net)

        self.learn_step_counter = 0  # for target updating
        self.memory_size = memory_size
        self.memory = ReplayMemory(self.memory_size)  # s,s_,reward,action
        self.memory_counter = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

        if not os.path.exists('./result'):
            os.makedirs('./result')

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        actions_value = self.eval_net.forward(x)
        action = actions_value.argmax().item()
        return action

    def store_transition(self, transition):
        self.memory.push(transition)
        if self.memory_counter < self.memory_size:
            self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')
        self.learn_step_counter += 1

        # sample batch transitions
        transitions = self.memory.sample(self.batch_size)
        batch = self.Transition(*zip(*transitions))

        b_s = torch.FloatTensor(batch.state)
        b_a = torch.LongTensor(batch.action).view(-1, 1)
        b_r = torch.FloatTensor(batch.reward).view(-1, 1)
        b_s_ = torch.FloatTensor(batch.next_state)
        b_d = torch.FloatTensor(batch.done).view(-1, 1)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.batch_size, 1)*(1-b_d)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.eval_net.reset_noise()
        self.target_net.reset_noise()

    def save_param(self):
        torch.save(self.target_net.state_dict(), 'result/target_net_params.pkl')
        torch.save(self.eval_net.state_dict(), 'result/eval_net_params.pkl')

