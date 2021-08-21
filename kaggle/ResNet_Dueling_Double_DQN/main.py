from kaggle_environments import make
import numpy as np
from DQNAgent import D3QN
import torch
from collections import namedtuple
from utils import *
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, translate

ACTIONS = ['NORTH', 'SOUTH', 'WEST', 'EAST']
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'validmoves'))

def main():
    env = make("hungry_geese")
    env.run()
    trainer = env.train([None, "greedy", "greedy", "greedy"])

    from tensorboardX import SummaryWriter
    writer = SummaryWriter()

    np.random.seed(0)
    torch.manual_seed(0)

    memory_size = 1000000
    batch_size = 32
    episodes = 30000

    agent = D3QN(s_dim=17, batch_size=batch_size, memory_size=memory_size, writer=writer)
    # agent.load_param(2)

    for episode in range(episodes):
        total_reward = 0
        obs = trainer.reset()
        last_obs = None

        while True:
            state = make_input(obs, last_obs)

            validmoves = getValidMoves(obs, last_obs, index=0)
            if episode < 1000:
                action = agent.choose_action(state, validmoves)
            else:
                action = agent.choose_action(state, validmoves, deterministic=True)

            next_obs, env_reward, done, info = trainer.step(ACTIONS[action])
            reward = get_rewards(env_reward, next_obs, obs, done)

            if done:
                trans = Transition(state, action, reward, state, done, validmoves)
            else:
                trans = Transition(state, action, reward, make_input(next_obs, obs), done, validmoves)
            agent.store_transition(trans)

            if agent.memory_counter >= 1000:
                agent.train(Transition)

            last_obs = obs
            obs = next_obs
            total_reward += reward

            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}. epsiode: {}'.format(
                    total_reward, next_obs['step'], reward, episode))
                if writer:
                    writer.add_scalar('Return', reward, episode)
                break

        agent.save_param(3)


if __name__ == '__main__':
    main()