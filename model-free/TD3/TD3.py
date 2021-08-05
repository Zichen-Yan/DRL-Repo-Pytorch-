import argparse
from collections import namedtuple
import numpy as np
from TD3_Agent import TD3
import gym
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'is_terminals'])


def train(env_name,
          seed,
          render,
          eval,
          tau,
          lr,
          gamma,
          episodes,
          batch_size,
          buffer_size,
          policy_noise,
          noise_clip,
          policy_freq,
          start_timestep,
          expl_noise,
          noise_decay
          ):
    env = gym.make(env_name)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    action_range = [env.action_space.low, env.action_space.high]
    # print(action_range)
    # print(env.observation_space)
    # print(env.action_space.shape[0])
    # raise

    agent = TD3(env=env,
                env_name=env_name,
                gamma=gamma,
                tau=tau,
                batch_size=batch_size,
                buffer_size=buffer_size,
                policy_noise=policy_noise,
                noise_clip=noise_clip,
                policy_freq=policy_freq,
                lr=lr
                )
    if eval:
        agent.load_param()
        print('Load Success!')

    sum_reward = []
    counter = 0
    for episode in range(episodes):
        ep_r = 0
        state = env.reset()
        for i in range(1000):
            if render:
                env.render()
            if counter <= start_timestep and not eval:
                counter += 1
                action = env.action_space.sample()
            else:
                action = agent.action(state)
                if not eval:
                    action += np.random.normal(0, env.action_space.high[0] * expl_noise,
                                                                    size=env.action_space.shape[0])
                expl_noise *= noise_decay
                action = action.clip(env.action_space.low, env.action_space.high)
            next_state, reward, done, _ = env.step(action)
            if reward <= -100:
                reward = -1
            trans = Transition(state, action, reward, next_state, done)
            agent.store_transition(trans)
            ep_r += reward
            if done:
                break
            if agent.memory_counter > start_timestep and not eval:
                agent.train()
            state = next_state
        print('Ep: ', episode, '| Ep_r: ', round(ep_r, 2))
        print(agent.memory_counter)
        sum_reward.append(ep_r)
        if ep_r >= 100 and not eval:
            agent.save_param()

    plt.plot(sum_reward)
    plt.ylabel('Return')
    plt.xlabel("Episode")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Env
    parser.add_argument("--env_name", default="FetchReach-v1")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)
    # Train
    parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
    parser.add_argument('--buffer_size', default=1000000, type=int)  # replay buffer size
    parser.add_argument('--episodes', default=3000, type=int)  # num of  games
    parser.add_argument('--batch_size', default=256, type=int)  # mini batch size
    parser.add_argument('--start_timestep', default=25000, type=int)

    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_freq', default=2, type=float)
    parser.add_argument('--expl_noise', default=0.25, type=float)
    parser.add_argument('--noise_decay', default=0.999, type=float)

    args = parser.parse_args()

    train(env_name=args.env_name,
          seed=args.seed,
          render=args.render,
          eval=args.eval,
          tau=args.tau,
          lr=args.learning_rate,
          gamma=args.gamma,
          episodes=args.episodes,
          batch_size=args.batch_size,
          buffer_size=args.buffer_size,
          policy_noise=args.policy_noise,
          noise_clip=args.noise_clip,
          policy_freq=args.policy_freq,
          start_timestep=args.start_timestep,
          expl_noise=args.expl_noise,
          noise_decay=args.noise_decay
          )