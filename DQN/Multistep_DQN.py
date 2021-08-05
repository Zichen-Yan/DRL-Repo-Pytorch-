import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import random
import os
import math
from copy import deepcopy
from utils import *

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, N_ACTIONS)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.out(x)

class DQN(object):
    EPS_START = 0.9  # greedy policy
    EPS_END = 0.05
    EPS_DECAY = 200

    GAMMA = 0.9  # reward discount
    TARGET_REPLACE_ITER = 100  # target update frequency

    def __init__(self, num_state, num_action, batch_size=32, lr=1e-3, memory_size=2000, n_step=3, Transition=None):
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

        self.n_step_buffer = n_step_buffer(n_step=n_step)
        self.n_step = n_step

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
        self.n_step_buffer.push(transition)
        if len(self.n_step_buffer.get_sample()) == self.n_step:
            transition_list = self.n_step_buffer.get_sample()
            batch = self.Transition(*zip(*transition_list))
            n_step_s = batch.state
            n_step_s_ = batch.next_state
            n_step_a = batch.action
            n_step_r = batch.reward
            n_step_d = batch.done

            discounted_n_step_r = [np.power(self.GAMMA, i) * r for i, r in enumerate(n_step_r)]
            discounted_n_step_r_sum = np.sum(discounted_n_step_r)
            n_step_state = n_step_s_[-1]
            cur_state = n_step_s[0]
            done = n_step_d[-1]
            action = n_step_a[0]

            self.memory.push((discounted_n_step_r_sum, n_step_state, cur_state, done, action))
            if self.memory_counter < self.memory_size:
                self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')
        self.learn_step_counter += 1

        # sample batch transitions
        minibatch = self.memory.sample(self.batch_size)

        minibatch = np.array(minibatch)
        discounted_n_step_r_sum = np.stack(minibatch[:, 0])
        n_step_state = np.stack(minibatch[:, 1])
        cur_state = np.stack(minibatch[:, 2])
        done = np.stack(minibatch[:, 3])
        action = np.stack(minibatch[:, 4])

        gamma = self.GAMMA ** self.n_step

        n_s_ = torch.FloatTensor(n_step_state)
        b_r = torch.FloatTensor(discounted_n_step_r_sum).view(-1, 1)
        b_d = torch.LongTensor(done).view(-1, 1)
        n_s = torch.FloatTensor(cur_state)
        n_a = torch.LongTensor(action).view(-1, 1)
        with torch.no_grad():
            q_next = self.target_net(n_s_)
            q_target = b_r + (1-b_d)*gamma*q_next.max(1)[0].view(self.batch_size, 1)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(n_s).gather(1, n_a)
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
