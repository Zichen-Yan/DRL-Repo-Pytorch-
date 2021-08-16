from collections import namedtuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical

class Actor(nn.Module):
    def __init__(self, num_action, num_state, hidden_layer=128, action_bound=1.0):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc_mu = nn.Linear(hidden_layer, num_action)
        self.fc_std = nn.Linear(hidden_layer, num_action)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.action_bound = action_bound

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.action_bound*self.tanh(self.fc_mu(x))
        std = self.softplus(self.fc_std(x)) + 1e-3
        return mu, std

class Critic(nn.Module):
    def __init__(self, num_state, hidden_layer=128):
        super(Critic, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(num_state, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class PPO(object):
    gamma = 0.95
    clip_param = 0.2
    max_grad_norm = 0.5
    gae_lambda = 0.95

    def __init__(self,
                 num_state,
                 num_action,
                 a_lr=1e-4,
                 c_lr=3e-4,
                 use_GAE=False,
                 action_bound=1.0):
        super(PPO, self).__init__()
        self.actor_net = Actor(num_state=num_state, num_action=num_action, action_bound=action_bound)
        self.critic_net = Critic(num_state=num_state)
        self.buffer = []
        self.training_step = 0
        self.use_GAE = use_GAE

        self.n_a = num_action
        self.n_s = num_state

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=a_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=c_lr)
        if not os.path.exists('result'):
            os.makedirs('result')

    def action(self, state):
        state = torch.from_numpy(state).float().view(1, -1)
        mu, std = self.actor_net(state)
        dist = Normal(mu, std)
        action = dist.sample()

        return action.detach().numpy().flatten(), dist.log_prob(action).detach().numpy().flatten()

    def evaluate(self, state, action):
        mu, std = self.actor_net(state)
        dist = Normal(mu, std)
        action_log_prob = dist.log_prob(action)  # ln(f(x))
        dist_entropy = dist.entropy()

        value = self.critic_net(state)
        return torch.squeeze(value), action_log_prob, dist_entropy

    def get_gaes(self, rewards, v_preds, v_preds_next, dones):
        deltas = [r_t + self.gamma * (1 - done) * v_next - v for r_t, v_next, v, done in zip(rewards, v_preds_next,
                                                                                             v_preds, dones)]
        advantages = []
        adv = 0.0
        for i in reversed(range(len(deltas))):
            adv = self.gae_lambda * self.gamma * adv * (1 - dones[i]) + deltas[i]
            advantages.append(adv)
        advantages.reverse()
        adv = torch.tensor(advantages, dtype=torch.float32)
        returns = adv + v_preds
        return adv, returns

    def dis_rewards(self, rewards, dones, next_state):
        discounted_reward = np.zeros_like(rewards)
        rollout_length = len(rewards)

        for t in reversed(range(rollout_length)):
            if t == rollout_length - 1:
                discounted_reward[t] = rewards[t] + self.gamma * (1 - dones[t]) * self.critic_net(next_state[-1])
            else:
                discounted_reward[t] = rewards[t] + self.gamma * (1 - dones[t]) * discounted_reward[t + 1]
        return torch.tensor(discounted_reward, dtype=torch.float32)

    def update(self):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float32)
        next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float32)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.int64).view(-1, self.n_a)
        reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float32).flatten()
        done = torch.tensor([t.is_terminals for t in self.buffer], dtype=torch.int64).flatten()
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).detach()

        #-----------------------------------------
        # A=r+gamma*v_-v  GAE gae_lambda=0
        # pred_v = self.critic_net(state).flatten()
        # pred_v_ = self.critic_net(next_state).flatten()
        # with torch.no_grad():
        #     value_target = reward + self.gamma * pred_v_ * (1 - done)
        #     adv = value_target - pred_v
        #     Gt = value_target
        #     adv = adv.view(-1, 1)

        # -----------------------------------------
        with torch.no_grad():
            if not self.use_GAE: # A=Gt-V GAE gae_lambda=1
                Gt = self.dis_rewards(reward, done, next_state)
                adv = Gt - torch.squeeze(self.critic_net(state))
            else:  # GAE gae_lambda=0.95
                # Gt = self.dis_rewards(reward, done, next_state)
                v = self.critic_net(state)
                next_v = self.critic_net(next_state)
                adv, Gt = self.get_gaes(reward, torch.squeeze(v), torch.squeeze(next_v), done)
            # normalize is optional
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv = adv.view(-1, 1)

        for i in range(10):
            V, new_action_log_prob, dist_entropy = self.evaluate(state, action)
            # Surrogate Loss
            ratio = torch.exp(new_action_log_prob - old_action_log_prob)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
            aloss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
            vloss = 0.5 * F.mse_loss(Gt, V)

            self.actor_optimizer.zero_grad()
            aloss.backward()
            nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            vloss.backward()
            nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            self.training_step += 1

        del self.buffer[:]  # clear experience

    def save_param(self):
        torch.save(self.actor_net.state_dict(), 'result/actor_net_params.pkl')
        torch.save(self.critic_net.state_dict(), 'result/critic_net_params.pkl')

    def load_params(self):
        self.actor_net.load_state_dict(torch.load('./result/actor_net_params.pkl'))
        self.critic_net.load_state_dict(torch.load('./result/critic_net_params.pkl'))

    def store_transition(self, transition):
        self.buffer.append(transition)

if __name__ == '__main__':
    agent = PPO(2, 2)
    pass
