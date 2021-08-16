import ray
from ray import tune
import time

ray.shutdown()
ray.init(ignore_reinit_error=True)

# 如果想使用pytorch框架，添加  'framework':'torch',

# num_workers指在多少个cpu核心上采集数据，如果不知道自己的机器有多少核心，通过下面的命令查看

# ray.available_resources()['CPU']

# ray.available_resources()['GPU']

# 注意：如果有24个核心，num_workers只能最大写到23，因为trainer占用一个核心，rolloutworker最多只能有23个，rolloutworker代表采集数据的worker，例如蒙特卡洛采样函数

# 通过我观察，num_workers并不是越大越快，其实不加num_workers参数，ray会自动分配，速度相比最大的核心数反而要快

# num_gpus对于单卡机器，可以设为分数，如0.5，代表当前的trainer占用一半的gpu资源，后面会看一下什么是trainer

config = {
    'env': 'CartPole-v1',
    'num_gpus': 1,
    'framework': 'torch',
    'num_workers': 7,
    "vf_share_layers": tune.grid_search([True, False]),
    "lr": tune.grid_search([1e-4, 1e-5, 1e-6]),
}

stop = {
    # 'episode_reward_mean': 500,
    'timesteps_total': 100000
}

st = time.time()

results = tune.run(
    'PPO',  # Specify the algorithm to train
    config=config,
    stop=stop
)

print('elapsed time=', time.time() - st)

ray.shutdown()