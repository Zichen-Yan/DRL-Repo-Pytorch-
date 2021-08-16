from torch.distributions.normal import Normal
from model import ActorCritic
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import ray
from copy import deepcopy

class Learner(object):
    clip_param = 0.2
    max_grad_norm = 0.5

    def __init__(self, args, shared_storage, replay_buffer, summary_writer, agent):
        self.args = args
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer
        self.summary_writer = summary_writer

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Running Learner on Device: {}".format(self.device))

        self.agent = agent.to(self.device)
        self.ac_optimizer = optim.Adam(self.agent.parameters(), lr=args.lr)

    def train_network(self):
        while ray.get(self.shared_storage.get_update_counter.remote()) < self.args.max_training_step:
            while ray.get(self.shared_storage.get_sample_counter.remote()) < self.args.worker_number:
                pass
            print(ray.get(self.replay_buffer.__len__.remote()))
            for i in range(self.args.training_steps):
                batch = ray.get(self.replay_buffer.sample.remote(self.args.batch_size))
                self.update_gradients(batch)
                ray.get(self.shared_storage.add_update_counter.remote())

            ray.get(self.shared_storage.set_weights.remote(self.agent.to("cpu").state_dict()))
            self.agent.to(self.device)

            ray.get(self.replay_buffer.empty.remote())
            ray.get(self.shared_storage.reset_sample_counter.remote())

    def evaluate(self, state, action):
        mu, std, value = self.agent(state)
        dist = Normal(mu, std)
        action_log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return torch.squeeze(value), action_log_prob, dist_entropy

    def update_gradients(self, batch):
        with torch.no_grad():
            state = torch.stack([t[0] for t in batch]).float().to(self.device)
            action = torch.stack([t[1] for t in batch]).float().view(-1, self.args.n_a).to(self.device)
            returns = torch.stack([t[2] for t in batch]).float().to(self.device)
            adv = torch.stack([t[3] for t in batch]).float().view(-1, 1).to(self.device)
            old_action_log_prob = torch.stack([t[4] for t in batch]).float().to(self.device)

        V, new_action_log_prob, dist_entropy = self.evaluate(state, action)

        # Surrogate Loss
        ratio = torch.exp(new_action_log_prob - old_action_log_prob)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
        aloss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
        vloss = 0.5 * F.mse_loss(returns, V)

        loss = aloss + vloss
        self.ac_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.ac_optimizer.step()
