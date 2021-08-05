import torch
import gym
from collections import namedtuple
import random
import seaborn as sns
import matplotlib.pyplot as plt
from vanilla_DQN import DQN

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def train(env_name='CartPole-v0', render=False, episodes=400, batch_size=32, load=False, memory_size=1000):
    env = gym.make(env_name).unwrapped
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.n
    agent = DQN(batch_size=batch_size, num_state=num_state, num_action=num_action, memory_size=int(memory_size), Transition=Transition)
    if load:
        agent.target_net.load_state_dict(torch.load('./result/target_net_params.pkl'))
        agent.eval_net.load_state_dict(torch.load('./result/eval_net_params.pkl'))
    sum_reward = []
    for episode in range(episodes):
        s = env.reset()
        ep_r = 0
        counter = 0
        if hasattr(agent, 'n_step_buffer'):
            agent.n_step_buffer.reset()

        while True:
            if render:
                env.render()
            a = agent.choose_action(s)

            s_, r, done, info = env.step(a)

            # reward shaping for CartPole
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            trans = Transition(s, a, r, s_, done)
            agent.store_transition(trans)
            ep_r += r
            if agent.memory_counter >= batch_size:
                agent.learn()
            if done or counter>=200:
                print('Ep: ', episode, '| Ep_r: ', round(ep_r, 2))
                break
            s = s_
            counter += 1
        sum_reward.append(ep_r)

    # dqn.save_param()
    plt.figure()
    sns.lineplot(x=range(len(sum_reward)), y=sum_reward)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--memory_size', type=int, default=1e5)
    args = parser.parse_args()
    print('\nGo!\n')
    train(env_name=args.env_name, render=args.render, episodes=args.episodes, batch_size=args.batch_size,
          load=args.load, memory_size=args.memory_size)
