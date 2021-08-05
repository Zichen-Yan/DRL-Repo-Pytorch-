import numpy as np
from cost_function import trajectory_cost_fn


class Controller():
    def __init__(self):
        pass

    def get_action(self, state):
        pass

class RandomController(Controller):
    def __init__(self, env):
        super(RandomController, self).__init__()
        self.env = env

    def get_action(self, state):
        return self.env.action_space.sample()


class MPCcontroller(Controller):
    def __init__(self,
                 env,
                 dyn_model,
                 horizon=15,
                 cost_fun=None,
                 num_simulated_paths=10,
                 ):
        super(MPCcontroller, self).__init__()

        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fun
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """
        ob, obs, next_obs, acts, costs = [], [], [], [], []  # (horizon, num_simulated_paths, n_dim)
        [ob.append(state) for _ in range(self.num_simulated_paths)]
        for _ in range(self.horizon):
            act = []
            obs.append(ob)
            [act.append(self.env.action_space.sample()) for _ in range(self.num_simulated_paths)]
            acts.append(act)
            ob = self.dyn_model.predict(np.array(ob), np.array(act))
            next_obs.append(ob)  # (1000, 20)
        # np.array(obs).shape=(15, 1000, 20)
        costs = trajectory_cost_fn(self.cost_fn, np.array(obs), np.array(acts), np.array(next_obs))
        j = np.argmin(costs, )

        return acts[0][j]  # 0 represents first action