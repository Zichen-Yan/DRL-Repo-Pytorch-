import ray
import numpy as np
import random
import gym
from collections import deque
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
from training import train, test
from argument import get_args


def timer(start, end):
    """ Helper to print training time """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == "__main__":
    args = get_args()
    # if training
    if not args.eval:
        ray.init()
        writer = SummaryWriter("runs/"+args.env)
        t0 = time.time()
        trained_model = train(args, writer)
        t1 = time.time()
        time.sleep(1.5)
        timer(t0, t1)
        # save model
        torch.save(trained_model, "result/"+args.env + ".pth")
    else:
        model_path = 'result/'+args.env+'.pth'
        test(args, model_path)
