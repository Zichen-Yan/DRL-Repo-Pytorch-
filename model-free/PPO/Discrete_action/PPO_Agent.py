from collections import namedtuple
import os
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import copy

class ActorCritic(nn.Module):
	def __init__(self, num_state, num_action, n_latent_var=64):
		super(ActorCritic, self).__init__()
		self.fc1 = nn.Linear(num_state, n_latent_var)
		self.fc2 = nn.Linear(n_latent_var, n_latent_var)
		self.fc3_pi = nn.Linear(n_latent_var, num_action)
		self.fc3_v = nn.Linear(n_latent_var, 1)
		self.relu = nn.ReLU(inplace=True)
		self.softmax = nn.Softmax(dim=-1)

	def actor(self, x):
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.fc3_pi(x)
		return self.softmax(x)

	def critic(self, x):
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.fc3_v(x)
		return x

class PPO(object):
	gamma = 0.99
	clip_param = 0.2
	max_grad_norm = 0.5  # 0.5
	gae_lambda = 0.95

	def __init__(self, num_state, num_action, lr=3e-4, use_GAE=False):
		super(PPO, self).__init__()
		self.ac_net = ActorCritic(num_state=num_state, num_action=num_action)
		self.buffer = []
		self.training_step = 0

		self.ac_optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)
		self.scheduler = optim.lr_scheduler.StepLR(self.ac_optimizer, step_size=5, gamma=0.999)
		if not os.path.exists('./result'):
			os.makedirs('./result')

		self.use_GAE = use_GAE

	def action(self, state, eval):
		state = torch.from_numpy(state).float().unsqueeze(0)
		action_prob = self.ac_net.actor(state)
		if not eval:
			dist = Categorical(probs=action_prob)
			action = dist.sample()
			return action.item(), dist.log_prob(action)
		else:
			return torch.argmax(action_prob).item, 0

	def evaluate(self, state, action):
		action_probs = self.ac_net.actor(state)
		dist = Categorical(probs=action_probs)
		action_log_prob = dist.log_prob(action)
		dist_entropy = dist.entropy()

		value = self.ac_net.critic(state)
		return torch.squeeze(value), action_log_prob, dist_entropy, action_probs.gather(1, action.unsqueeze(0))

	def get_gaes(self, rewards, v_preds, v_preds_next, dones):
		deltas = [r_t + self.gamma * (1 - done) * v_next - v for r_t, v_next, v, done in zip(rewards, v_preds_next,
																							 v_preds, dones)]
		advantages = []
		adv = 0.0
		for d in deltas[::-1]:
			adv = self.gae_lambda * self.gamma * adv + d
			advantages.append(adv)
		advantages.reverse()
		adv=torch.tensor(advantages, dtype=torch.float32)
		returns=adv+v_preds
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

	def update(self):
		state = torch.tensor([t.state for t in self.buffer], dtype=torch.float32).detach()
		next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float32).detach()
		action = torch.tensor([t.action for t in self.buffer], dtype=torch.int64).detach()
		reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float32).flatten()
		done = torch.tensor([t.is_terminals for t in self.buffer], dtype=torch.int64).flatten()
		old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).detach()

		with torch.no_grad():
			if not self.use_GAE:
				Gt = self.dis_rewards(reward, done, next_state)
				# optional
				# Gt = (Gt - Gt.mean()) / (Gt.std() + 1e-8)
				adv = Gt - torch.squeeze(self.ac_net.critic(state))
			else:
				# Gt = self.dis_rewards(reward, done, next_state)
				v = self.ac_net.critic(state)
				next_v = self.ac_net.critic(next_state)
				adv, Gt = self.get_gaes(reward, torch.squeeze(v), torch.squeeze(next_v), done)
				# normalize is optional
				# adv = (adv - adv.mean()) / (adv.std() + 1e-8)

		for i in range(50):
			V, action_log_prob, dist_entropy, action_prob = self.evaluate(state, action)
			# Surrogate Loss
			ratio = torch.exp(action_log_prob - old_action_log_prob)
			surr1 = ratio * adv
			surr2 = torch.clamp(ratio, 1. - self.clip_param, 1. + self.clip_param) * adv
			aloss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
			vloss = F.mse_loss(Gt, V)
			loss = aloss + 0.5*vloss

			self.ac_optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
			self.ac_optimizer.step()
			self.scheduler.step()

			KL = torch.squeeze(action_prob) * (action_log_prob - old_action_log_prob)
			if KL.mean() >= 0.005:
				# print(KL.mean())
				break

			self.training_step += 1
		del self.buffer[:]  # clear experience

	def save_param(self):
		torch.save(self.ac_net.state_dict(), 'result/actor_net_params.pkl')

	def load_param(self):
		self.ac_net.load_state_dict(torch.load('result/actor_net_params.pkl'))

	def store_transition(self, transition):
		self.buffer.append(transition)


if __name__ == '__main__':
	agent = PPO(2, 2)
	pass
