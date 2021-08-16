import fire
from collections import deque, namedtuple
from tqdm import tqdm
import numpy as np
from torch.distributions import Categorical

import gym

Memory = namedtuple('Memory', ['state', 'action', 'action_log_prob', 'reward', 'done', 'value'])

from utils import *
from PPGagent import PPG


def train(
        env_name='LunarLander-v2',
        num_episodes=50000,
        max_timesteps=500,
        actor_hidden_dim=32,
        critic_hidden_dim=256,
        minibatch_size=64,
        lr=0.0005,
        betas=(0.9, 0.999),
        lam=0.95,
        gamma=0.99,
        eps_clip=0.2,
        value_clip=0.4,
        beta_s=.01,
        update_timesteps=5000,
        num_policy_updates_per_aux=32,
        epochs=1,
        epochs_aux=6,
        seed=None,
        render=False,
        save_every=1000,
        load=False,
):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    memories = deque([])
    aux_memories = deque([])

    agent = PPG(
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        epochs_aux,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        value_clip
    )

    if load:
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    num_policy_updates = 0

    for eps in tqdm(range(num_episodes), desc='episodes'):
        state = env.reset()
        for timestep in range(max_timesteps):
            time += 1
            if render:
                env.render()

            action, action_log_prob, value = agent.action(state)
            next_state, reward, done, _ = env.step(action)

            memory = Memory(state, action, action_log_prob, reward, done, value)
            memories.append(memory)

            state = next_state

            if time % update_timesteps == 0:
                agent.learn(memories, aux_memories, next_state)
                num_policy_updates += 1
                memories.clear()

                if num_policy_updates % num_policy_updates_per_aux == 0:
                    agent.learn_aux(aux_memories)
                    aux_memories.clear()

            if done:
                break

        if eps % save_every == 0:
            agent.save()


if __name__ == '__main__':
    train()
