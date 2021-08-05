from collections import namedtuple
import matplotlib.pyplot as plt
from DDPG_Agent import DDPG
import gym
import seaborn as sns
import numpy as np

# Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

def train(env_name='Pendulum-v0', render=False, eval=False, episodes=1000, memory_size=100000):
    env = gym.make(env_name).unwrapped
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.shape[0]
    a_bound = env.action_space.high
    a_low_bound = env.action_space.low

    agent = DDPG(n_s=num_state, n_a=num_action, memory_size=memory_size, a_bound=a_bound)
    var = 3
    if eval:
        agent.load_params()
        print('Load Success!')
        var = 0
    sum_reward = []
    counter = 1

    for episode in range(episodes):
        state = env.reset()
        ep_r = 0
        for t in range(500):
            if render:
                env.render()

            action = agent.action(state)
            a = np.clip(np.random.normal(action, var), a_low_bound, a_bound)
            next_state, reward, done, _ = env.step(a)
            ep_r += reward
            agent.store_transition(state, a, reward / 10, next_state)
            # trans = Transition(state, action, reward/10, next_state)
            # agent.store_transition(trans)

            if agent.pointer > memory_size:
                var *= 0.9995  # decay the exploration controller factor
                if ~eval:
                    agent.learn()
                counter = 0
            counter += 1
            if done:
                break
            state = next_state

        print('Ep: ', episode, '| Ep_r: ', round(ep_r, 2))
        if ep_r >= 100 and ~eval:
            agent.save_param()
        sum_reward.append(ep_r)
    plt.figure()
    sns.lineplot(x=range(len(sum_reward)), y=sum_reward)
    plt.xlabel("episode")
    plt.ylabel("reward")

    plt.figure()
    sns.lineplot(x=range(len(agent.TD_loss)), y=agent.TD_loss)
    plt.xlabel("episode")
    plt.ylabel("TD_loss")
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--render', action='store_true')
    # parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--memory_size', type=int, default=30000)
    parser.add_argument('--eval', type=bool, default=False)
    args = parser.parse_args()
    print('\nGo!\n')
    train(env_name=args.env_name, render=args.render, episodes=args.episodes, eval=args.eval,
          memory_size=args.memory_size)