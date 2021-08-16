import os
import sys
import gym
import time
from collections import deque, namedtuple
import numpy as np
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import torch
import torch.nn.functional as F
import torch.optim as optim
from model import ActorCritic
from copy import deepcopy
from utils import ReplayMemory

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'is_terminals'])


class PPO_worker:
    gamma = 0.99
    clip_param = 0.2
    max_grad_norm = 0.5

    def __init__(self, rank, args, traffic_signal=None, ac_counter=None,
                 shared_model=None, shared_obs_filter=None, shared_grad_buffers=None):
        self.args = args
        self.env = gym.make(args.env_name).unwrapped
        self.use_GAE = args.use_gae
        self.gae_lambda = args.gae_lambda

        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)

        self.buffer = ReplayMemory(args.buffer_size)

        # get the numbers of observation and actions...
        self.n_s = self.env.observation_space.shape[0]
        if self.env.action_space.shape:
            self.n_a = self.env.action_space.shape[0]
        else:
            self.n_a = self.env.action_space.n
        act_bound = self.env.action_space.high[0]
        # define the network...
        self.ac_net = ActorCritic(self.n_s, self.n_a, action_bound=act_bound)
        self.old_ac_net = deepcopy(self.ac_net)
        # shared object
        self.traffic_signal = traffic_signal
        self.ac_counter = ac_counter
        self.shared_model = shared_model
        self.shared_obs_filter = shared_obs_filter
        self.shared_grad_buffers = shared_grad_buffers
        print('Load workers')

    def action(self, state):
        state = torch.from_numpy(state).float().view(1, -1)
        mu, std = self.ac_net.actor(state)
        dist = Normal(mu, std)
        action = dist.sample()

        return action.detach().numpy().flatten(), dist.log_prob(action).detach().numpy().flatten()

    def evaluate(self, state, action):
        mu, std, value = self.ac_net(state)
        dist = Normal(mu, std)
        action_log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return torch.squeeze(value), action_log_prob, dist_entropy

    def get_gaes(self, rewards, v_preds, v_preds_next, dones):
        deltas = [r_t + self.gamma * (1 - done) * v_next - v for r_t, v_next, v, done in zip(rewards, v_preds_next,
                                                                                             v_preds, dones)]
        advantages = []
        adv = 0.0
        for i in reversed(range(len(deltas))):
            adv = self.gae_lambda * self.gamma * adv * (1-dones[i]) + deltas[i]
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
                discounted_reward[t] = rewards[t] + self.gamma * (1 - dones[t]) * self.ac_net.critic(next_state[-1])
            else:
                discounted_reward[t] = rewards[t] + self.gamma * (1 - dones[t]) * discounted_reward[t + 1]
        return torch.tensor(discounted_reward, dtype=torch.float32)

    def train_network(self):
        while True:
            self.ac_net.load_state_dict(self.shared_model.state_dict())

            reward_sum = 0
            buffer = []
            state = self.env.reset()
            state = self.shared_obs_filter.normalize(state)

            for step in range(self.args.collection_length):
                # self.env.render()
                action, action_prob = self.action(state)
                next_state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                next_state = self.shared_obs_filter.normalize(next_state)

                trans = Transition(state, action, action_prob, reward / 10.0, next_state, done)
                buffer.append(trans)
                state = next_state

                if done:
                    reward_sum = 0
                    state = self.env.reset()
                    state = self.shared_obs_filter.normalize(state)

            # start to calculate the gradients for this time sequence...
            # reward_buffer.add(reward_sum / self.args.collection_length)
            self.preprocess_buffer(buffer)
            self.update_network()

    def preprocess_buffer(self, buffer):
        state = torch.tensor([t.state for t in buffer], dtype=torch.float32)
        next_state = torch.tensor([t.next_state for t in buffer], dtype=torch.float32)
        action = torch.tensor([t.action for t in buffer], dtype=torch.float32).view(-1, self.n_a)
        reward = torch.tensor([t.reward for t in buffer], dtype=torch.float32).flatten()
        done = torch.tensor([t.is_terminals for t in buffer], dtype=torch.float32).flatten()
        old_action_log_prob = torch.tensor([t.a_log_prob for t in buffer], dtype=torch.float)

        with torch.no_grad():
            # -----------------------------------------
            # A=r+gamma*v_-v  GAE gae_lambda=1
            # pred_v = self.critic_net(state).flatten()
            # pred_v_ = self.critic_net(next_state).flatten()

            # value_target = reward + self.gamma * pred_v_ * (1 - done)
            # adv = value_target - pred_v
            # Gt = value_target

            if not self.use_GAE:  # A=Gt-V GAE gae_lambda=0
                Gt = self.dis_rewards(reward, done, next_state)
                adv = Gt - torch.squeeze(self.ac_net.critic(state))
            else:  # GAE gae_lambda=0.95
                # Gt = self.dis_rewards(reward, done, next_state)
                v = self.ac_net.critic(state)
                next_v = self.ac_net.critic(next_state)
                adv, Gt = self.get_gaes(reward, torch.squeeze(v), torch.squeeze(next_v), done)
            # normalize is optional
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            for s, a, returns, adv, a_log in zip(state, action, Gt, adv, old_action_log_prob):
                self.buffer.push([s, a, returns, adv, a_log])


    def update_network(self):
        for i in range(self.args.ac_update_step):
            self.ac_net.load_state_dict(self.shared_model.state_dict())
            self.ac_net.zero_grad()
            signal_init = self.traffic_signal.get()

            batch = self.buffer.sample(self.args.batch_size)

            state = torch.stack([t[0] for t in batch]).float()
            action = torch.stack([t[1] for t in batch]).float().view(-1, self.n_a)
            returns = torch.stack([t[2] for t in batch]).float()
            adv = torch.stack([t[3] for t in batch]).float().view(-1, 1).detach()
            old_action_log_prob = torch.stack([t[4] for t in batch]).float().detach()

            V, new_action_log_prob, dist_entropy = self.evaluate(state, action)

            # Surrogate Loss
            ratio = torch.exp(new_action_log_prob - old_action_log_prob)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
            aloss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
            vloss = 0.5 * F.mse_loss(returns.detach(), V)

            loss = aloss + vloss
            loss.backward()
            # nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
            self.shared_grad_buffers.add_gradient(self.ac_net)
            self.ac_counter.increment()

            while signal_init == self.traffic_signal.get():
                pass

        self.buffer.empty()

    def test_network(self, model_path):
        # load the models and means and std...
        policy_model, running_mean_filter = torch.load(model_path, map_location=lambda storage, loc: storage)
        mean = running_mean_filter[0]
        std = running_mean_filter[1]

        self.ac_net.load_state_dict(policy_model)
        self.ac_net.eval()

        # start to test...
        while True:
            state = self.env.reset()
            reward_sum = 0
            while True:
                self.env.render()
                state = self.normalize_filter(state, mean, std)
                state = torch.tensor(state).float().view(1, -1)
                mu, _ = self.ac_net.actor(state)
                next_state, reward, done, _ = self.env.step(mu.detach().numpy().flatten())
                # sum the reward...
                reward_sum += reward
                if done:
                    break
                state = next_state

            print('the reward sum in this episode is ' + str(reward_sum) + '!')

    def normalize_filter(self, x, mean, std):
        x = (x - mean) / (std + 1e-8)
        x = np.clip(x, -5.0, 5.0)

        return x

def test(rank, args, shared_model, shared_obs_filter):
    torch.manual_seed(args.seed + rank)
    env = gym.make(args.env_name).unwrapped

    n_s = env.observation_space.shape[0]
    if env.action_space.shape:
        n_a = env.action_space.shape[0]
    else:
        n_a = env.action_space.n
    action_bound = env.action_space.high[0]

    model = ActorCritic(n_s, n_a, action_bound)

    state = env.reset()
    sum_r = 0
    max_r = 0
    steps = 0
    while True:
        model.load_state_dict(shared_model.state_dict())
        # env.render()
        steps += 1
        state = shared_obs_filter.normalize(state)
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu, std = model.actor(state)
        dist = Normal(mu, std)
        action = dist.sample()
        state, reward, done, _ = env.step(action.detach().numpy().flatten())
        sum_r += reward

        if done or steps >= args.max_episode_length:
            print("episode reward {}, episode length {}".format(sum_r, steps))
            if sum_r >= max_r:
                save_path = 'result/' + args.env_name + '_model.pt'
                torch.save([shared_model.state_dict(), shared_obs_filter.get_results()], save_path)
            max_r = max(max_r, sum_r)
            sum_r = 0
            steps = 0
            state = env.reset()
            time.sleep(10)

