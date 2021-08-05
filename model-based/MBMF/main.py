from collections import namedtuple
import gym
from controllers import RandomController, MPCcontroller
import numpy as np
import torch
from utils import sample, compute_normalization
from dynamics import NNDynamicsModel
from cheetah_env import HalfCheetahEnvNew
from cost_function import cheetah_cost_fn
from MPC_net import MPC_net
import matplotlib.pyplot as plt
import seaborn as sns
from Agent.DDPG_Agent import DDPG
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'is_terminals'])


def train(env_name,
          seed,
          render=False,
          lr=1e-3,
          onpolicy_iters=10,
          dyn_iters=60,
          batch_size=512,
          random_paths=10,
          onpolicy_paths=10,
          simulated_paths=10000,
          mpc_horizon=15,
          env_horizon=1000,
          eval=False,
          episodes=1000,
          episode_step=1000
          ):

    part = 1
    train_net = 'DDPG'

    np.random.seed(seed)
    torch.manual_seed(seed)

    if env_name == "HalfCheetah-v3":
        env = HalfCheetahEnvNew()
        cost_fn = cheetah_cost_fn
    else:
        env = gym.make(env_name).unwrapped
    random_controller = RandomController(env)
    data = sample(env, random_controller, random_paths, env_horizon)
    normalization = compute_normalization(data)

    mpc_net = MPC_net(env, normalization=normalization, env_name=env_name)

    dyn_model = NNDynamicsModel(env=env,
                                env_name=env_name,
                                batch_size=batch_size,
                                iterations=dyn_iters,
                                learning_rate=lr,
                                normalization=normalization
                                )
    mpc_controller = MPCcontroller(env=env,
                                   dyn_model=dyn_model,
                                   horizon=mpc_horizon,
                                   cost_fun=cost_fn,
                                   num_simulated_paths=simulated_paths
                                   )

    if part == 1:  # train dynamics_net
        for itr in range(onpolicy_iters):
            dyn_model.fit(data)
            paths = sample(env, mpc_controller, onpolicy_paths, env_horizon)
            data = np.concatenate((data, paths))
        dyn_model.save_params()
    if part == 2:  # collect data for mpc_net
        dyn_model.load_params()
        paths = sample(env, mpc_controller, num_paths=20, horizon=1000)
        r = np.concatenate([d["reward"] for d in paths])
        print(np.mean(r))

        s = np.concatenate([d["state"] for d in paths])
        a = np.concatenate([d["action"] for d in paths])  # (timestep, 2)

        if not os.path.exists('./data'):
            os.makedirs('./data')
        np.savez(os.path.join('/data', env_name + '_train_data_for_mpcnet'), state=s, action=a)

    if part == 3:
        mpc_net.load_params()
        if train_net == 'DDPG':
            agent = DDPG(env)
            var = 2
            if eval:
                agent.load_params()
                print('Load Success!')
                var = 0
            sum_reward = []
            counter = 1
            counter_mpc = 0

            for episode in range(episodes):
                state = env.reset()
                ep_r = 0
                for t in range(episode_step):
                    if render:
                        env.render()
                    if counter_mpc <= 100000 and not eval:
                        _, _, action = mpc_net.get_action(state)
                        counter_mpc += 1
                    else:
                        action = agent.action(state)
                        action = np.clip(np.random.normal(action, var), env.action_space.low, env.action_space.high)
                        var *= 0.9995
                    next_state, reward, done, _ = env.step(action)
                    ep_r += reward
                    agent.store_transition(state, action, reward, next_state)

                    if not eval and agent.pointer > 2000:
                        agent.learn()
                    if done:
                        break
                    state = next_state

                if ep_r >= 1000 and not eval:
                    agent.save_param()
                sum_reward.append(ep_r)
                print('Ep: ', episode, '| Ep_r: ', round(ep_r, 2))

            plt.figure()
            sns.lineplot(x=range(len(agent.TD_loss)), y=agent.TD_loss)
            plt.xlabel("episode")
            plt.ylabel("TD_loss")

        plt.figure()
        sns.lineplot(x=range(len(sum_reward)), y=sum_reward)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--env_name', type=str, default="HalfCheetah-v3")
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)
    # Training args
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--onpolicy_iters', type=int, default=2)
    parser.add_argument('--dyn_iters', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--episodes', type=int, default=150)
    parser.add_argument('--episode_step', type=int, default=1000)
    # Data collection
    parser.add_argument('--random_paths', type=int, default=10)
    parser.add_argument('--onpolicy_paths', type=int, default=10)  # 10
    parser.add_argument('--simulated_paths', type=int, default=1000)  # 1000
    parser.add_argument('--env_horizon', type=int, default=1000)
    # MPC
    parser.add_argument('--mpc_horizon', type=int, default=15)
    args = parser.parse_args()
    print('\nGo!\n')
    train(env_name=args.env_name,
          seed=args.seed,
          render=args.render,
          lr=args.lr,
          onpolicy_iters=args.onpolicy_iters,
          dyn_iters=args.dyn_iters,
          batch_size=args.batch_size,
          random_paths=args.random_paths,
          onpolicy_paths=args.onpolicy_paths,
          simulated_paths=args.simulated_paths,
          env_horizon=args.env_horizon,
          mpc_horizon=args.mpc_horizon,
          eval=args.eval,
          episodes=args.episodes,
          episode_step=args.episode_step
          )