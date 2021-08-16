import torch
import torch.multiprocessing as mp
import random
import numpy as np
from copy import deepcopy

class TrafficLight:
    """used by chief to allow workers to run or not"""

    def __init__(self, val=True):
        self.val = mp.Value("b", False)
        self.lock = mp.Lock()

    def get(self):
        with self.lock:
            return self.val.value

    def switch(self):
        with self.lock:
            self.val.value = (not self.val.value)

class Counter:
    """enable the chief to access worker's total number of updates"""

    def __init__(self):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()

    def get(self):
        # used by chief
        with self.lock:
            return self.val.value

    def increment(self):
        # used by workers
        with self.lock:
            self.val.value += 1

    def reset(self):
        # used by chief
        with self.lock:
            self.val.value = 0


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, trans):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[int(self.position)] = trans
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def empty(self):
        self.memory = []
        self.position = 0

class Shared_grad_buffers:
    def __init__(self, models):
        self.grads = {}
        for name, p in models.named_parameters():
            self.grads[name + '_grad'] = torch.zeros(p.size()).share_memory_()

    def add_gradient(self, models):
        for name, p in models.named_parameters():
            self.grads[name + '_grad'] += p.grad.data

    def reset(self):
        for name, grad in self.grads.items():
            self.grads[name].fill_(0)

class RewardCounter:
    def __init__(self):
        self.val = mp.Value('f', 0)
        self.cnt = mp.Value('i', 0)
        self.lock = mp.Lock()

    def add(self, reward):
        with self.lock:
            self.cnt.value += 1
            self.val.value += reward

    def get(self):
        with self.lock:
            return self.val.value/(self.cnt.value+1e-8)

    def reset(self):
        with self.lock:
            self.val.value = 0
            self.cnt.value = 0

class Running_mean_filter:
    def __init__(self, num_inputs):
        self.lock = mp.Lock()
        self.n = torch.zeros(num_inputs).share_memory_()
        self.mean = torch.zeros(num_inputs).share_memory_()
        self.s = torch.zeros(num_inputs).share_memory_()
        self.var = torch.zeros(num_inputs).share_memory_()
    # start to normalize the states...
    def normalize(self, x):
        with self.lock:
            obs = x.copy()
            obs = torch.Tensor(obs)
            self.n += 1
            if self.n[0] == 1:
                self.mean[...] = obs
                self.var[...] = self.mean.pow(2)
            else:
                old_mean = self.mean.clone()
                self.mean[...] = old_mean + (obs - old_mean) / self.n
                self.s[...] = self.s + (obs - old_mean) * (obs - self.mean)
                self.var[...] = self.s / (self.n - 1)
            mean_clip = self.mean.numpy().copy()
            var_clip = self.var.numpy().copy()
            std = np.sqrt(var_clip)
            x = (x - mean_clip) / (std + 1e-8)
            x = np.clip(x, -5.0, 5.0)
            return x
    # start to get the results...
    def get_results(self):
        with self.lock:
            var_clip = self.var.numpy().copy()
            return (self.mean.numpy().copy(), np.sqrt(var_clip))