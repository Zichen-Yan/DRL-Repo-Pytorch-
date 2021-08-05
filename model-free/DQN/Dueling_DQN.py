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

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 64)
        self.fc2 = nn.Linear(64, 64)
        self.adv = nn.Linear(64, N_ACTIONS)
        self.value=nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        adv=self.adv(x)
        avgadv = torch.mean(adv, dim=1, keepdim=True)

        v=self.value(x)
        return v+adv-avgadv

class DQN(object):
    EPS_START = 0.9  # greedy policy
    EPS_END = 0.05
    EPS_DECAY = 200

    GAMMA = 0.9  # reward discount
    TARGET_REPLACE_ITER = 100  # target update frequency

    def __init__(self, num_state, num_action, batch_size=32, lr=1e-3, memory_size=2000, Transition=None):
        super(DQN, self).__init__()
        self.n_a = num_action
        self.n_s = num_state
        self.batch_size = batch_size
        self.Transition = Transition

        self.eval_net = Net(N_STATES=num_state, N_ACTIONS=num_action)
        self.target_net = deepcopy(self.eval_net)

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
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.memory_counter / self.EPS_DECAY)

        if np.random.uniform(0, 1) > eps_threshold:  # greedy
            actions_value = self.eval_net.forward(x)
            action = actions_value.argmax().item()  # return the argmax index
        else:  # random
            action = np.random.randint(0, self.n_a)
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

        # b_r = (b_r - b_r.mean()) / (b_r.std() + 1e-7)
        # print(b_r)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.batch_size, 1)*(1-b_d)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_param(self):
        torch.save(self.target_net.state_dict(), 'result/target_net_params.pkl')
        torch.save(self.eval_net.state_dict(), 'result/eval_net_params.pkl')

if __name__ == '__main__':
    agent = DQN(2, 2)
    pass
