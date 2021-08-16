from arguments import get_args
from worker import PPO_worker

# achieve the arguments
args = get_args()

# build th worker...
env = args.env_name
worker = PPO_worker(rank=0, args=args)
model_path = 'result/'+env+'_model.pt'
worker.test_network(model_path)
