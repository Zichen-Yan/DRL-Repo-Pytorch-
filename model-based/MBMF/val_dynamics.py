from controllers import RandomController
import numpy as np
from utils import sample, compute_normalization
from dynamics import NNDynamicsModel
from cheetah_env import HalfCheetahEnvNew
from cost_function import cheetah_cost_fn
import matplotlib.pyplot as plt

env = HalfCheetahEnvNew()
cost_fn = cheetah_cost_fn

random_controller = RandomController(env)
data = sample(env, random_controller, horizon=100, num_paths=1)
normalization = compute_normalization(data)

dyn_model = NNDynamicsModel(env=env,
                            batch_size=512,
                            iterations=60,
                            learning_rate=1e-3,
                            normalization=normalization
                            )
dyn_model.load_params()

s = np.concatenate([d["state"] for d in data])  # (timestep, 8)
a = np.concatenate([d["action"] for d in data])
s_ = np.concatenate([d["next_state"] for d in data])

pre_s=[]
obs = [s[0,:]]
for i in range(100):
    obs=dyn_model.predict(np.array(obs), np.array([a[i,:]]))
    pre_s.append(obs)

pre_s = np.array(pre_s).reshape(100,20)

t=np.linspace(0,100,100)

plt.plot(t, s_[:,5], linewidth=0.4, linestyle='-')
plt.plot(t, pre_s[:,5], linewidth=0.4, linestyle='--')
# plt.plot(t, pre_s, color='blue', linewidth=0.4, linestyle='.')
plt.show()
