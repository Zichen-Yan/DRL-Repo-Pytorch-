import torch
import numpy as np
from storage import SharedStorage
from replay_buffer import ReplayBuffer
import gym
from model import ActorCritic
from learner import Learner
from worker import Worker
import ray
from copy import deepcopy

@ray.remote
def evaluation(args, shared_storage, agent):

    # build environment
    env = gym.make(args.env).unwrapped
    env.seed(args.seed)

    agent.eval()
    his = 0
    with torch.no_grad():
        while ray.get(shared_storage.get_update_counter.remote()) < args.max_training_step:
            x = ray.get(shared_storage.get_update_counter.remote())
            if x % args.checkpoint_interval == 0 and x!=his:
                # run eval game
                his = x
                agent.set_weights(ray.get(shared_storage.get_weights.remote()))
                counter = ray.get(shared_storage.get_interactions.remote())
                state = env.reset()
                rewards = 0

                while True:
                    state = torch.from_numpy(state).float().view(1, -1)
                    mu, std = agent.actor(state)
                    state, reward, done, _ = env.step(mu.detach().numpy().flatten())
                    rewards += reward
                    if done:
                        break
                print("Interaction Steps: {} | Evaluation Reward: {:.2f} | Learning Steps: {} ".format(counter, rewards, x))
                # shared_storage.set_eval_reward.remote(counter, step, rewards)


def train(args, writer):
    # build environment
    env = gym.make(args.env)
    n_s = env.observation_space.shape[0]
    n_a = env.action_space.shape[0]
    args.n_a = n_a
    action_bound = env.action_space.high[0]

    agent = ActorCritic(n_s, n_a, action_bound)

    # initialize storage and replay buffer
    storage = SharedStorage.remote(agent)
    replay_buffer = ReplayBuffer.remote(args.buffer_size)

    # create a number of distributed worker 
    workers = [Worker.remote(worker_id, args, storage, replay_buffer, deepcopy(agent)) for worker_id in range(0, args.worker_number)]
    # collect experience
    for w in workers:
        w.run.remote()
    # add evaluation worker 
    evals = [evaluation.remote(args, storage, deepcopy(agent))]
    # Create Learner
    learner = Learner(args, storage, replay_buffer, writer, agent)
    learner.train_network()

    ray.get(evals)
    return ray.get(storage.get_weights.remote())


def test(args, model_path):
    env = gym.make(args.env).unwrapped
    n_s = env.observation_space.shape[0]
    n_a = env.action_space.shape[0]

    action_high = env.action_space.high[0]
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    # create actor based on args details
    actor = ActorCritic(n_s, n_a, action_bound=action_high)

    # load_weights
    actor.load_state_dict(torch.load(model_path))
    actor.eval()

    while True:
        state = env.reset()
        rewards = 0
        while True:
            env.render()
            state = torch.from_numpy(state).float().view(1, -1)
            mu, _ = actor.actor(state)
            state, reward, done, _ = env.step(mu.detach().numpy().flatten())
            rewards += reward
            if done:
                break
        print("\nTest Rewards: {}".format(rewards))