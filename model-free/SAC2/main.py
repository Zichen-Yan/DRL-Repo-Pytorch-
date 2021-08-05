import gym
import numpy as np
import itertools
import torch
from sac import SAC
from replay_memory import ReplayMemory
from utils import evalution
from argument import get_args


def train(args):
    # Environment
    env = gym.make(args.env_name).unwrapped

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = SAC(env, args)
    if args.eval:
        agent.load_model(args.env_name)
    # Memory
    memory = ReplayMemory(args.replay_size)

    # Training Loop
    total_numsteps = 0
    eval_reward_list = []
    for episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False

        state = env.reset()
        while not done:
            if args.render:
                env.render()
            if total_numsteps <= args.start_steps and not args.eval:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state, eval=args.eval)  # Sample action from policy

            if len(memory) > args.batch_size and not args.eval:
                agent.update_parameters(memory, args.batch_size)

            next_state, reward, done, _ = env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            done_mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, done_mask)  # Append transition to memory
            state = next_state

        print("Episode: {}, total numsteps: {}, current steps: {}, reward: {}".format(episode, total_numsteps,
                                                                                     episode_steps, round(episode_reward, 2)))

        if episode % 15 == 0 and not args.eval:
            avg_reward = evalution(env, agent)
            eval_reward_list.append(avg_reward)
            if avg_reward >= 4000:
                agent.save_model(args.env_name)

        if total_numsteps > args.num_steps:  # 1000000
            break


    import matplotlib.pyplot as plt
    plt.plot(eval_reward_list)
    plt.ylabel('Return')
    plt.xlabel("Episode")
    plt.grid(True)
    plt.show()

    env.close()


if __name__ == '__main__':
    args = get_args()
    train(args)

