from controllers import RandomController, MPCcontroller
import numpy as np
from utils import sample, compute_normalization
from dynamics import NNDynamicsModel
from cheetah_env import HalfCheetahEnvNew
from cost_function import cheetah_cost_fn
from MPC_net import MPC_net
import matplotlib.pyplot as plt

env = HalfCheetahEnvNew()
cost_fn = cheetah_cost_fn

random_controller = RandomController(env)
data = sample(env, random_controller, horizon=100, num_paths=1)
normalization = compute_normalization(data)

dyn_model = NNDynamicsModel(env=env,
                            env_name='HalfCheetah-v3',
                            batch_size=512,
                            iterations=60,
                            learning_rate=1e-3,
                            normalization=normalization
                            )
dyn_model.load_params()

mpc_controller = MPCcontroller(env=env,
                               dyn_model=dyn_model,
                               horizon=15,
                               cost_fun=cost_fn,
                               num_simulated_paths=1000
                               )

mpc_net = MPC_net(env, normalization=normalization, env_name='HalfCheetah-v3')
mpc_net.load_params()

data = sample(env, mpc_controller, num_paths=1, horizon=100)

s = np.concatenate([d["state"] for d in data])
a = np.concatenate([d["action"] for d in data])
s_ = np.concatenate([d["next_state"] for d in data])

pre_act = []
for i in range(100):
    _, _, act = mpc_net.get_action(np.array([s[i,:]]))
    pre_act.append(act)

pre_act = np.array(pre_act).reshape(100, -1)

t=np.linspace(0,100,100)

plt.plot(t, a[:,1], linewidth=0.4, linestyle='-')
plt.plot(t, pre_act[:,1], linewidth=0.4, linestyle='--')
# plt.plot(t, pre_s, color='blue', linewidth=0.4, linestyle='.')
plt.show()
