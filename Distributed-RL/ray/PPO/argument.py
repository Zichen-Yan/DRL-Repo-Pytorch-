import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLanderContinuous-v2")
    parser.add_argument("--training_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer_size", type=int, default=int(1e5))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval", type=str, default=True, help="Load a saved model to perform a test run!")
    parser.add_argument("--worker_number", type=int, default=4)
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument("--max_training_step", type=int, default=10000)

    parser.add_argument('--use_gae', type=bool, default=True)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    args = parser.parse_args()

    return args
