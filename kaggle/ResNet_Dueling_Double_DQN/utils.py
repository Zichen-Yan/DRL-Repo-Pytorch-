import numpy as np
from copy import deepcopy

import torch
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, \
    Configuration, Action, row_col, adjacent_positions, translate, min_distance, random_agent, GreedyAgent


def centerize(b):
    dx, dy = np.where(b[0])
    centerize_x = (np.arange(0, 7) - 3 + dx[0]) % 7
    centerize_y = (np.arange(0, 11) - 5 + dy[0]) % 11

    b = b[:, centerize_x, :]
    b = b[:, :, centerize_y]

    return b


def make_input(obs, last_obs):
    b = np.zeros((17, 7 * 11), dtype=np.float32)

    for p, pos_list in enumerate(obs['geese']):
        # head position
        for pos in pos_list[:1]:
            b[0 + (p - obs['index']) % 4, pos] = 1
        # tip position
        for pos in pos_list[-1:]:
            b[4 + (p - obs['index']) % 4, pos] = 1
        # whole position
        for pos in pos_list:
            b[8 + (p - obs['index']) % 4, pos] = 1
    # previous head position
    if last_obs is not None:
        for p, pos_list in enumerate(last_obs['geese']):
            for pos in pos_list[:1]:
                b[12 + (p - obs['index']) % 4, pos] = 1

    # food
    for pos in obs['food']:
        b[16, pos] = 1

    b = b.reshape(-1, 7, 11)
    b = centerize(b)

    return b

def get_rank(obs, prev_obs):
    geese = obs['geese']
    index = obs['index']
    player_len = len(geese[index])
    survivors = [i for i in range(len(geese)) if len(geese[i]) > 0]
    if index in survivors:  # if our player survived in the end, its rank is given by its length in the last state
        return sum(len(x) >= player_len for x in geese)  # 1 is the best, 4 is the worst
    # if our player is dead, consider lengths in penultimate state
    geese = prev_obs['geese']
    index = prev_obs['index']
    player_len = len(geese[index])
    rank_among_lost = sum(len(x) >= player_len for i, x in enumerate(geese) if i not in survivors)
    return rank_among_lost + len(survivors)


def get_rewards(env_reward, obs, prev_obs, done):
    geese = prev_obs['geese']
    index = prev_obs['index']
    step = prev_obs['step']
    cur_goose = obs['geese'][0]
    if done:
        rank = get_rank(obs, prev_obs)
        r1 = (10, 5, -1, -5)[rank - 1]
        died_from_hunger = ((step + 1) % 40 == 0) and (len(geese[index]) == 1)
        r2 = -10 if died_from_hunger else 0  # int(rank == 1) # huge penalty for dying from hunger and huge award for the win
        r3 = -20 if env_reward == 0 else 0
        reward = r1 + r2 + r3
    else:
        if env_reward == 100 or env_reward == 201 or env_reward == 202 or env_reward==99:
            reward = 1
            if len(cur_goose) == 1:
                reward = -1
        elif env_reward == 101:
            reward = 40
        else:
            print(env_reward)
            raise Exception('Strange Reward')

    return reward


def num_to_pos(num):
    head = num
    x = int(head / 11)
    y = head - 11 * x
    return [x, y]


def surround_pos(x, y):
    left = (y - 1) % 11
    right = (y + 1) % 11
    up = (x - 1) % 7
    down = (x + 1) % 7
    surround_pos = [[up, y], [down, y], [x, left], [x, right]]
    return surround_pos


def getValidMoves(obs, last_obs=None, index=0):
    geese = obs.geese
    # --------------------------------------------------------
    head_poses = []
    for i in geese:
        if len(i) > 0:
            head_pos = num_to_pos(i[0])
        else:
            head_pos = []
        head_poses.append(head_pos)

    [head_x, head_y] = head_poses[index]
    sur_points = surround_pos(head_x, head_y)
    if last_obs is not None:
        last_head_pos = num_to_pos(last_obs.geese[index][0])
        sur_points.remove(last_head_pos)

    dangerous_pos = []
    for pos in sur_points:
        sub_sur_points = surround_pos(pos[0], pos[1])
        num = 0
        for head in head_poses:
            if head in sub_sur_points:
                num += 1
        if num >= 2:
            dangerous_pos.append(pos)
    masks = []
    if len(dangerous_pos) > 0:
        for i in dangerous_pos:
            [f_x, f_y] = i
            if f_x - head_x == -1 or f_x - head_x == 6:
                mask = [0., 1., 1., 1.]
            elif f_x - head_x == 1 or f_x - head_x == -6:
                mask = [1., 0., 1., 1.]
            elif f_x - head_x == 0:
                if f_y - head_y == -1 or f_y - head_y == 10:
                    mask = [1., 1., 0., 1.]
                elif f_y - head_y == 1 or f_y - head_y == -10:
                    mask = [1., 1., 1., 0.]
            masks.append(mask)
    # --------------------------------------------------------
    cur_pos = geese[index][0]
    obstacles = [position for goose in geese for position in goose[:-1]]
    if last_obs is not None:
        obstacles.append(last_obs.geese[index][0])
    valid_moves = [
        translate(cur_pos, action, 11, 7) not in obstacles
        for action in [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]
    ]
    for i in range(4):
        if not valid_moves[i]:
            valid_moves[i] = 0.
        else:
            valid_moves[i] = 1.
    if not any(valid_moves):
        valid_moves = [1., 1., 1., 1.]

    # --------------------------------------------------------
    my_tail = geese[index][-1]
    obstacles = [position for goose in geese for position in goose[-2:]]
    if my_tail in obstacles:
        obstacles.remove(my_tail)

    dangerous_moves = [
        translate(cur_pos, action, 11, 7) not in obstacles
        for action in [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]
    ]

    for i in range(4):
        if not dangerous_moves[i]:
            dangerous_moves[i] = 0.
        else:
            dangerous_moves[i] = 1.
    if not any(dangerous_moves):
        dangerous_moves = [1., 1., 1., 1.]
    # --------------------------------------------------------
    if len(masks) > 0:
        for mask in masks:
            dangerous_moves = (np.array(dangerous_moves) * np.array(mask)).tolist()
    # --------------------------------------------------------
    final_moves = (np.array(dangerous_moves) * np.array(valid_moves)).tolist()

    if not any(final_moves):
        final_moves = valid_moves
    return valid_moves
