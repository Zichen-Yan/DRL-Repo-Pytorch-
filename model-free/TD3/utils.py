import random
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'is_terminals'])

def generate_goals(i, episode_cache, sample_num, sample_range=50):
    end = (i+sample_range) if i+sample_range < len(episode_cache) else len(episode_cache)
    epi_to_go = episode_cache[i:end]
    if len(epi_to_go) < sample_num:
        sample_trans = epi_to_go
    else:
        sample_trans = random.sample(epi_to_go, sample_num)
    new_goals = []
    for trans in sample_trans:
        mid = np.reshape(trans[3]['achieved_goal'], (1, 3))
        new_goals.append(mid)
    return new_goals


def rescale(state, s_dim):
    obs = np.reshape(state['observation'], (1, s_dim))
    des = np.reshape(state['desired_goal'], (1, 3))
    return np.concatenate([obs, des], axis=1)


def update_trans(new_goals, transition, env):
    state, next_state = transition[0], transition[3]
    action = transition[1]
    info = transition[4]
    done = transition[5]
    # update state
    obs_dim = env.observation_space['observation'].shape[0]
    state_ = np.concatenate((state['observation'].reshape(1, obs_dim), new_goals.reshape(1, 3)), axis=1)
    next_state_ = np.concatenate((next_state['observation'].reshape(1, obs_dim), new_goals.reshape(1, 3)), axis=1)

    # compute reward
    reward = env.compute_reward(state['achieved_goal'].reshape(1, -1), new_goals, info)

    trans = Transition(state_, action, reward.item(), next_state_, done)

    return trans
