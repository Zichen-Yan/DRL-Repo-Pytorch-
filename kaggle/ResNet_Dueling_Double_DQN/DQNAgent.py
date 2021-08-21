import numpy as np
import random
import torch
import torch.nn as nn
from copy import deepcopy
from Qnet import Qnet


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, trans):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = trans
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class D3QN(object):
    EPSILON = 0.1  # greedy policy
    GAMMA = 0.8  # reward discount
    TARGET_REPLACE_ITER = 150  # target update frequency
    tau = 0.005

    def __init__(self, s_dim=17, a_dim=4, batch_size=32, lr=1e-4, memory_size=2000, writer=None):
        super(D3QN, self).__init__()
        self.a_dim = a_dim
        self.batch_size = batch_size
        self.writer = writer

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.eval_net = Qnet(dim_in=s_dim, dim_out=self.a_dim).to(self.device)
        self.target_net = deepcopy(self.eval_net)

        self.learn_step_counter = 0  # for target updating
        self.memory_size = memory_size
        self.memory = ReplayMemory(self.memory_size)  # s,s_,reward,action
        self.memory_counter = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, obs, validmoves, deterministic=False):
        x = torch.unsqueeze(torch.FloatTensor(obs), 0).to(self.device)
        if not deterministic:
            if np.random.uniform() < (1 - self.EPSILON):
                actions_value, value = self.eval_net.forward(x)
                actions_value = actions_value.cpu().data.numpy() * validmoves
                actions_value = np.where(actions_value == 0, -9999, actions_value)
                action = np.argmax(actions_value, 1)[0]
            else:
                action = np.random.randint(0, self.a_dim)
        else:
            actions_value, value = self.eval_net.forward(x)
            actions_value = actions_value.cpu().data.numpy() * validmoves
            actions_value = np.where(actions_value == 0, -9999, actions_value)
            action = np.argmax(actions_value, 1)[0]

        return action

    def store_transition(self, transition):
        self.memory.push(transition)
        if self.memory_counter < self.memory_size:
            self.memory_counter += 1

    def train(self, Transition):
        # if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())
        #     print('\ntarget_params_replaced\n')
        self.learn_step_counter += 1

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        b_s = torch.FloatTensor(batch.state).to(self.device)
        b_a = torch.LongTensor(batch.action).view(-1, 1).to(self.device)
        b_r = torch.FloatTensor(batch.reward).view(-1, 1).to(self.device)
        b_s_ = torch.FloatTensor(batch.next_state).to(self.device)
        b_validmoves = torch.FloatTensor(batch.validmoves).to(self.device)

        # q_eval w.r.t the action in experience
        q_eval, _ = self.eval_net(b_s)
        q_eval = q_eval.gather(1, b_a)  # shape (batch, 1)

        with torch.no_grad():
            q_next, _ = self.eval_net(b_s_)
            target_q_next, _ = self.target_net(b_s_)

            target_q_action = torch.argmax(q_next, dim=1, keepdim=True)
            q_target = b_r + self.GAMMA * target_q_next.gather(1, target_q_action)

        # invalid mask
        # q_next *= b_validmoves
        # for i in range(self.batch_size):
        #     for j in range(4):
        #         if abs(q_next[i][j]) <= 1e-5:
        #             q_next[i][j] = -9999

        loss = self.loss_func(q_eval, q_target)

        if self.writer:
            self.writer.add_scalar('loss', loss.detach(), self.learn_step_counter)
            self.writer.add_scalar('Q_eval', torch.mean(q_eval).detach(), self.learn_step_counter)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for param, target_param in zip(self.eval_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_param(self, index):
        torch.save(self.eval_net.state_dict(), 'eval_net_params' + str(index) + '.pkl')

    def load_param(self, index):
        self.eval_net.load_state_dict(torch.load('eval_net_params' + str(index) + '.pkl'))
        self.target_net = deepcopy(self.eval_net)
