from arguments import get_args
from worker import PPO_worker

# achieve the arguments
args = get_args()

# build th worker...
worker = PPO_worker(args)
model_path = '/home/yzc/Pycharm Projects/Pytorch_resources/Mine_RL_code/distributed_PPO/result/CartPole-v1_models_0.pt'
worker.test_network(model_path)
