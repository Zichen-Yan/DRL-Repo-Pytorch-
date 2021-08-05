import numpy as np
from cost_function import trajectory_cost_fn
import tqdm


def sample(env,
           controller,
           num_paths=10,
           horizon=1000,
           imitation=False
           ):
    paths = []
    for _ in tqdm.tqdm(range(num_paths)):
        state = env.reset()
        obs, next_obs, acts, rewards, costs = [], [], [], [], []
        steps = 0
        while True:
            obs.append(state)
            if not imitation:
                act = controller.get_action(state)
            else:
                _, _, act = controller.get_action(state)
            acts.append(act)
            state, r, done, _ = env.step(act)
            next_obs.append(state)
            rewards.append(r)
            steps += 1
            if done or steps >= horizon:
                break
        path = {"state": np.array(obs),
                "next_state": np.array(next_obs),
                "reward": np.array(rewards),
                "action": np.array(acts)}
        paths.append(path)

    return paths

def compute_normalization(data):
    s = np.concatenate([d["state"] for d in data])
    sp = np.concatenate([d["next_state"] for d in data])
    a = np.concatenate([d["action"] for d in data])

    mean_obs = np.mean(s, axis=0)
    mean_deltas = np.mean(sp - s, axis=0)
    mean_action = np.mean(a, axis=0)

    std_obs = np.std(s, axis=0)
    std_deltas = np.std(sp - s, axis=0)
    std_action = np.std(a, axis=0)

    return mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action

def path_cost(cost_fn, path):
    return trajectory_cost_fn(cost_fn, path['state'], path['action'], path['next_state'])

