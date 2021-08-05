from collections import namedtuple
import matplotlib.pyplot as plt
from PPO_Agent import PPO
import gym
import seaborn as sns
import torch
import numpy as np

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'is_terminals'])

def train(args):
	env = gym.make(args.env_name).unwrapped
	torch.manual_seed(args.seed)

	num_state = env.observation_space.shape[0]
	num_action = env.action_space.n
	agent = PPO(num_state=num_state, num_action=num_action, use_GAE=args.use_GAE)
	if args.eval:
		agent.load_param()
		print('Load Success!')
	sum_reward = []
	counter = 0
	for episode in range(args.episodes):
		state = env.reset()
		ep_r = 0
		for t in range(201):
			if args.render:
				env.render()
			counter += 1
			action, action_prob = agent.action(state, args.eval)
			next_state, reward, done, _ = env.step(action)
			ep_r += reward

			trans = Transition(state, action, action_prob, reward/10.0, next_state, done)
			agent.store_transition(trans)

			if counter % 32 == 0:
				if not args.eval:
					agent.update()

			if done:
				break
			state = next_state
		print('Ep: ', episode, '| Ep_r: ', round(ep_r, 2))

		sum_reward.append(ep_r)
		# if ep_r >= 180 and not args.eval:
		# 	agent.save_param()

	plt.figure()
	sns.lineplot(x=range(len(sum_reward)), y=sum_reward)
	plt.xlabel("episode")
	plt.ylabel("reward")
	plt.show()


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	# parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
	parser.add_argument('--env_name', '--env', type=str, default='Acrobot-v1')

	parser.add_argument('--seed', type=int, default=123)
	parser.add_argument('--render', type=bool, default=False)
	parser.add_argument('--episodes', type=int, default=300)
	parser.add_argument('--eval', type=bool, default=False)
	parser.add_argument('--use_GAE', type=bool, default=True)
	args = parser.parse_args()
	train(args)
