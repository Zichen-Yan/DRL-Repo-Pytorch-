import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

LR_ACTOR = 0.001
LR_CRITIC = 0.002
GAMMA = 0.9
TAU = 0.01

class ActorNet(nn.Module):
    def __init__(self, num_action, num_state, hidden_layer=64, a_bound=1):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_layer)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(hidden_layer, num_action)
        self.fc3.weight.data.normal_(0, 0.1)
        self.a_bound = a_bound

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        actions = x * self.a_bound  # for the game "Pendulum-v0", action range is [-2, 2]
        return actions

class CriticNet(nn.Module):
    def __init__(self, num_action, num_state, hidden_layer=64):
        super(CriticNet, self).__init__()
        self.s_to_hidden = nn.Linear(num_state, hidden_layer)
        self.s_to_hidden.weight.data.normal_(0, 0.1)
        self.a_to_hidden = nn.Linear(num_action, hidden_layer)
        self.a_to_hidden.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_layer, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = self.s_to_hidden(s)
        y = self.a_to_hidden(a)
        actions_value = self.out(F.relu(x + y))
        return actions_value

class DDPG(object):
    def __init__(self, env, memory_size=100000, batch_size=128):
        self.a_dim, self.s_dim = env.action_space.shape[0], env.observation_space.shape[0]
        self.memory = np.zeros((memory_size, self.s_dim * 2 + self.a_dim + 1), dtype=np.float32)
        self.pointer = 0  # serves as updating the memory data
        # Create the 4 network objects
        self.actor_eval = ActorNet(self.a_dim, self.s_dim)
        self.actor_target = ActorNet(self.a_dim, self.s_dim)
        self.critic_eval = CriticNet(self.a_dim, self.s_dim)
        self.critic_target = CriticNet(self.a_dim, self.s_dim)
        # create 2 optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_CRITIC)
        # Define the loss function for critic network update
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.loss_func = nn.MSELoss()
        self.TD_loss=[]

        self.root = os.path.dirname(os.path.realpath(__file__))
        self.save_dir = os.path.join(self.root, "DDPG_net_param/")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def store_transition(self, s, a, r, s_):  # how to store the episodic data to buffer
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_size  # replace the old data with new data
        self.memory[index, :] = transition
        self.pointer += 1

    def action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        action = self.actor_eval(s)[0].detach()
        return action

    def learn(self):
        # softly update the target networks
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')
            # sample from buffer a mini-batch data
        indices = np.random.choice(self.memory_size, size=self.batch_size)
        batch_trans = self.memory[indices, :]
        # extract data from mini-batch of transitions including s, a, r, s_
        batch_s = torch.FloatTensor(batch_trans[:, :self.s_dim])
        batch_a = torch.FloatTensor(batch_trans[:, self.s_dim:self.s_dim + self.a_dim])
        batch_r = torch.FloatTensor(batch_trans[:, -self.s_dim - 1: -self.s_dim])
        batch_s_ = torch.FloatTensor(batch_trans[:, -self.s_dim:])
        # make action and evaluate its action values
        a = self.actor_eval(batch_s)
        q = self.critic_eval(batch_s, a)
        actor_loss = -torch.mean(q)
        # optimize the loss of actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # compute the target Q value using the information of next state
        a_target = self.actor_target(batch_s_)
        q_tmp = self.critic_target(batch_s_, a_target)
        q_target = batch_r + GAMMA * q_tmp
        # compute the current q value and the loss
        q_eval = self.critic_eval(batch_s, batch_a)
        td_error = self.loss_func(q_target, q_eval)

        # optimize the loss of critic network
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()

        self.TD_loss.append(td_error.detach().numpy().tolist())

    def load_params(self):
        self.actor_eval.load_state_dict(torch.load(self.save_dir + '/actor_eval_net_params.pkl'))
        self.actor_target.load_state_dict(torch.load(self.save_dir + '/actor_target_net_params.pkl'))
        self.critic_eval.load_state_dict(torch.load(self.save_dir + '/critic_eval_net_params.pkl'))
        self.critic_target.load_state_dict(torch.load(self.save_dir + '/critic_target_net_params.pkl'))

    def save_param(self):
        torch.save(self.actor_eval.state_dict(), self.save_dir + '/actor_eval_net_params.pkl')
        torch.save(self.actor_target.state_dict(), self.save_dir + '/actor_target_net_params.pkl')
        torch.save(self.critic_eval.state_dict(), self.save_dir + '/critic_eval_net_params.pkl')
        torch.save(self.critic_target.state_dict(), self.save_dir + '/critic_target_net_params.pkl')




