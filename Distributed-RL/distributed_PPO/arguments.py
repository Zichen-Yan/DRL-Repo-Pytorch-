import argparse

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--seed', type=int, default=123, help='the random seed')
    parse.add_argument('--env_name', default='LunarLanderContinuous-v2', help='environments name')

    parse.add_argument('--ac_lr', type=float, default=1e-4, help='the learning rate of actor network')
    parse.add_argument('--batch_size', type=int, default=256, help='the batch size of the training')
    parse.add_argument('--ac_update_step', type=int, default=10, help='the update number of actor network')
    parse.add_argument('--max_episode_length', type=int, default=100, metavar='LENGTH', help='Maximum episode length')
    parse.add_argument('--collection_length', type=int, default=64, help='the sample collection length(episodes)')
    parse.add_argument('--buffer_size', type=int, default=1000, help='the sample collection length(episodes)')

    args = parse.parse_args()

    return args


