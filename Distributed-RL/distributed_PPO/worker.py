import os
import sys
import gym
import time
from collections import deque, namedtuple
import numpy as np
from torch.distributions import Normal, Categorical

import torch
import torch.nn.functional as F
import torch.optim as optim
from model import ActorCritic

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'is_terminals'])


class PPO_worker:
    gamma = 0.99
    clip_param = 0.2
    max_grad_norm = 0.5
    gae_lambda = 0.95
    use_GAE = False

    def __init__(self, rank, args, traffic_signal=None, ac_counter=None,
                 shared_model=None, shared_obs_filter=None, shared_grad_buffers=None, shared_reward=None):
        self.args = args
        self.env = gym.make(args.env_name).unwrapped

        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)

        self.buffer = []

        # get the numbers of observation and actions...
        self.n_s = self.env.observation_space.shape[0]
        if self.env.action_space.shape:
            self.n_a = self.env.action_space.shape[0]
        else:
            self.n_a = self.env.action_space.n
        # define the network...
        self.ac_net = ActorCritic(self.n_s, self.n_a)
        # shared object
        self.traffic_signal = traffic_signal
        self.ac_counter = ac_counter
        self.shared_model = shared_model
        self.shared_obs_filter = shared_obs_filter
        self.shared_grad_buffers = shared_grad_buffers
        self.shared_reward = shared_reward
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
        for d in deltas[::-1]:
            adv = self.gae_lambda * self.gamma * adv + d
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
        print('Train')
        self.ac_net.load_state_dict(self.shared_model.state_dict())
        while True:
            reward_sum = 0
            self.buffer = []
            state = self.env.reset()
            for step in range(self.args.collection_length):
                # self.env.render()
                state = self.shared_obs_filter.normalize(state)
                action, action_prob = self.action(state)
                next_state, reward, done, _ = self.env.step(action)
                reward_sum += reward

                trans = Transition(state, action, action_prob, reward/10.0, next_state, done)
                self.buffer.append(trans)

                state = next_state

                if done:
                    self.shared_reward.add(reward_sum)
                    reward_sum = 0
                    state = self.env.reset()

            # start to calculate the gradients for this time sequence...
            # reward_buffer.add(reward_sum / self.args.collection_length)

            self.update_network()

    def update_network(self):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float32)
        next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float32)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.int64).view(-1, self.n_a)
        reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float32).flatten()
        done = torch.tensor([t.is_terminals for t in self.buffer], dtype=torch.int64).flatten()
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).detach()

        # -----------------------------------------
        # A=r+gamma*v_-v  GAE gae_lambda=1
        # pred_v = self.critic_net(state).flatten()
        # pred_v_ = self.critic_net(next_state).flatten()
        # with torch.no_grad():
        #     value_target = reward + self.gamma * pred_v_ * (1 - done)
        #     adv = value_target - pred_v
        #     Gt = value_target
        #     adv = adv.view(-1, 1)

        with torch.no_grad():
            if not self.use_GAE:  # A=Gt-V GAE gae_lambda=0
                Gt = self.dis_rewards(reward, done, next_state)
                # optional
                # Gt = (Gt - Gt.mean()) / (Gt.std() + 1e-8)
                adv = Gt - torch.squeeze(self.ac_net.critic(state))
            else:  # GAE gae_lambda=0.95
                Gt = self.dis_rewards(reward, done, next_state)
                v = self.ac_net.critic(state)
                next_v = self.ac_net.critic(next_state)
                adv, Gt = self.get_gaes(reward, torch.squeeze(v), torch.squeeze(next_v), done)
                # normalize is optional
                # adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv = adv.view(-1, 1)

        for i in range(self.args.ac_update_step):
            # compute advantage
            self.ac_net.zero_grad()
            signal_init = self.traffic_signal.get()

            V, new_action_log_prob, dist_entropy = self.evaluate(state, action)

            # Surrogate Loss
            ratio = torch.exp(new_action_log_prob - old_action_log_prob)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
            aloss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
            vloss = 0.5 * F.mse_loss(Gt, V)

            loss = aloss + vloss
            loss.backward()
            # nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
            self.shared_grad_buffers.add_gradient(self.ac_net)
            self.ac_counter.increment()

            while signal_init == self.traffic_signal.get():
                pass
            self.ac_net.load_state_dict(self.shared_model.state_dict())

            # KL = F.kl_div(old_action_log_prob, torch.squeeze(action_prob), reduction='mean')
            # print('KL:',KL)
            # if KL >= 0.025:
            #     break
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
                action = self.action(state)
                next_state, reward, done, _ = self.env.step(action)
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

    model = ActorCritic(n_s, n_a)

    state = env.reset()
    sum_r = 0

    start_time = time.time()

    steps = 0
    model.load_state_dict(shared_model.state_dict())
    model.eval()

    while True:
        # env.render()
        steps += 1
        state = shared_obs_filter.normalize(state)
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu, std = model.actor(state)
        dist = Normal(mu, std)
        action = dist.sample()
        state, reward, done, _ = env.step(action.detach().numpy().flatten())
        sum_r += reward

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                sum_r, steps))
            sum_r = 0
            steps = 0
            state = env.reset()
            time.sleep(20)
