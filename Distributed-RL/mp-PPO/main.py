import arguments
import torch.optim as optim
import torch.multiprocessing as mp
import gym
import os
from utils import TrafficLight, Counter, Shared_grad_buffers, Running_mean_filter
from model import ActorCritic
from worker import test, PPO_worker
from chief import chief


def train(args):
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method("spawn")
    env = gym.make(args.env_name)

    n_s = env.observation_space.shape[0]
    if env.action_space.shape:
        n_a = env.action_space.shape[0]
    else:
        n_a = env.action_space.n
    action_bound = env.action_space.high[0]

    shared_model = ActorCritic(n_s, n_a, action_bound)
    shared_model.share_memory()

    shared_grad_buffers = Shared_grad_buffers(shared_model)
    traffic_light = TrafficLight()
    counter = Counter()
    shared_obs_filter = Running_mean_filter(n_s)

    optimizer = optim.Adam(shared_model.parameters(), lr=args.ac_lr)

    num_of_workers = mp.cpu_count() - 2
    processes = []
    workers = []

    p = mp.Process(target=test, args=(num_of_workers, args, shared_model, shared_obs_filter))
    processes.append(p)

    p = mp.Process(target=chief, args=(args, traffic_light, counter, num_of_workers,
                                       shared_model, shared_grad_buffers, optimizer, shared_obs_filter))
    processes.append(p)

    for idx in range(num_of_workers):
        workers.append(PPO_worker(idx, args, traffic_light, counter, shared_model,
                                  shared_obs_filter, shared_grad_buffers))

    for worker in workers:
        p = mp.Process(target=worker.train_network)
        processes.append(p)
    for p in processes:
        p.start()
    print('start')
    for p in processes:
        p.join()


if __name__ == '__main__':
    args = arguments.get_args()
    train(args)
