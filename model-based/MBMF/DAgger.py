from controllers import RandomController, MPCcontroller
import numpy as np
from utils import sample, compute_normalization
from dynamics import NNDynamicsModel
from cheetah_env import HalfCheetahEnvNew
from cost_function import cheetah_cost_fn
from MPC_net import MPC_net
import gym
import os

def train(env_name,
          mpc_horizon,
          num_paths,
          dagger_iter,
          env_horizon
         ):
    if env_name == "HalfCheetah-v3":
        env = HalfCheetahEnvNew()
        cost_fn = cheetah_cost_fn
    else:
        env = gym.make(env_name).unwrapped

    random_controller = RandomController(env)
    data = sample(env, random_controller, horizon=1000, num_paths=1)
    normalization = compute_normalization(data)

    dyn_model = NNDynamicsModel(env=env,
                                env_name=env_name,
                                batch_size=512,
                                iterations=60,
                                learning_rate=1e-3,
                                normalization=normalization
                                )
    dyn_model.load_params()

    mpc_controller = MPCcontroller(env=env,
                                   dyn_model=dyn_model,
                                   horizon=mpc_horizon,
                                   cost_fun=cost_fn,
                                   num_simulated_paths=1000
                                   )

    mpc_net = MPC_net(env, normalization=normalization, env_name=env_name)

    data = np.load('./data/' + env_name + '_train_data_for_mpcnet.npz')
    train_x = data['state']
    train_y = data['action']

    for i in range(dagger_iter):
        mpc_net.fit(s=train_x, a=train_y)
        paths = sample(env, mpc_net, num_paths=num_paths, horizon=env_horizon, imitation=True)
        s = np.concatenate([d["state"] for d in paths])
        r = np.concatenate([d["reward"] for d in paths])
        print('reward:', np.mean(r))

        acts = []
        for j in range(num_paths*env_horizon):
            act = mpc_controller.get_action(s[j, :])
            acts.append(act)
        new_acts = np.stack(np.array(acts))
        train_x = np.concatenate((train_x, s), 0)
        train_y = np.concatenate((train_y, new_acts), 0)

    mpc_net.save_params()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--env_name', type=str, default="HalfCheetah-v3")
    # Training args
    parser.add_argument('--dagger_iter', type=int, default=3)
    parser.add_argument('--mpc_horizon', type=int, default=15)
    parser.add_argument('--num_paths', type=int, default=5)
    parser.add_argument('--env_horizon', type=int, default=1000)
    args = parser.parse_args()
    print('\nGo!\n')
    train(env_name=args.env_name,
          env_horizon=args.env_horizon,
          mpc_horizon=args.mpc_horizon,
          dagger_iter=args.dagger_iter,
          num_paths=args.num_paths
          )