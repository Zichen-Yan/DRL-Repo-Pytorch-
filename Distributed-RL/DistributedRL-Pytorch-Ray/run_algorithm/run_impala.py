import ray
import torch
import time

from utils.utils import run_setting
from utils.environment import Environment
from utils.run_env import test_agent
from utils.replaybuffer import ImpalaBuffer
from utils.parameter_server import ParameterServer


from agents.algorithms.a2c_vtrace import A2CVtrace
from agents.runners.actors.impala_actor import ImpalaActor
from agents.runners.learners.impala_learner import ImpalaLearner

from copy import deepcopy

def run(args, agent_args):
    args, agent_args, env, state_dim, action_dim, writer, device = run_setting(args, agent_args)
    algorithm = A2CVtrace(writer, device, state_dim, action_dim, agent_args)
    
    learner = ImpalaLearner.remote()
    actors = [ImpalaActor.remote() for _ in range(args.num_actors)]
    buffer = ImpalaBuffer.remote(agent_args['learner_memory_size'], state_dim, 1, agent_args)
    
    learner.init.remote(deepcopy(algorithm), agent_args)
    ray.get([agent.init.remote(idx, deepcopy(algorithm), agent_args) for idx, agent in enumerate(actors)])
    ps = ParameterServer.remote(ray.get(learner.get_weights.remote()))
    
    [actor.run.remote(args.env_name, ps, buffer, args.epochs) for actor in actors]
    test_agent.remote(args.env_name, deepcopy(algorithm), ps, args.test_repeat, args.test_sleep)
    
    time.sleep(3)
    
    print('learner start')
    for epoch in range(args.epochs):
        ray.wait([learner.run.remote(ps, buffer)])
    print('learner finish')