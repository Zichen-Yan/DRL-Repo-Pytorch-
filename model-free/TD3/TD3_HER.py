import argparse
from collections import namedtuple
import numpy as np
from TD3_Agent_HER import TD3
import gym
import torch
import matplotlib.pyplot as plt
from utils import generate_goals, rescale, update_trans

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
          noise_decay,
          HER_sample_num
          ):
    env = gym.make(env_name)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space['observation'].shape[0]
    # action_range = [env.action_space.low, env.action_space.high]
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
        episode_cache = []
        for i in range(1000):  # 50 steps
            if render: env.render()
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
            next_state, reward, done, info = env.step(action)

            # save memory
            episode_cache.append((state, action, reward, next_state, info, done))
            trans = Transition(rescale(state, obs_dim), action, reward, rescale(next_state, obs_dim), done)
            agent.store_transition(trans)

            ep_r += reward
            if done:
                break
            state = next_state
        print(i)
        # save data
        sum_reward.append(ep_r)
        print('Ep: ', episode, '| Ep_r: ', round(ep_r, 2))
        print(agent.memory_counter)

        if not eval:
            if ep_r >= -30:
                agent.save_param()
            # HER
            for i, transition in enumerate(episode_cache):
                new_goals = generate_goals(i, episode_cache, HER_sample_num)
                for new_goal in new_goals:
                    new_trans = update_trans(new_goal, transition, env)
                    agent.store_transition(new_trans)
            # train
            if agent.memory_counter > start_timestep:
                for k in range(40):
                    agent.train()

    plt.plot(sum_reward)
    plt.ylabel('Return')
    plt.xlabel("Episode")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Env
    parser.add_argument("--env_name", default="FetchPickAndPlace-v1")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--eval', type=bool, default=True)
    # Train
    parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
    parser.add_argument('--buffer_size', default=1e6, type=int)  # replay buffer size
    parser.add_argument('--episodes', default=80000, type=int)  # num of  games
    parser.add_argument('--batch_size', default=128, type=int)  # mini batch size
    parser.add_argument('--start_timestep', default=25000, type=int)

    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_freq', default=2, type=float)
    parser.add_argument('--expl_noise', default=0.25, type=float)
    parser.add_argument('--noise_decay', default=0.999, type=float)
    # HER
    parser.add_argument('--HER_sample_num', default=4, type=int)

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
          noise_decay=args.noise_decay,
          HER_sample_num=args.HER_sample_num
          )