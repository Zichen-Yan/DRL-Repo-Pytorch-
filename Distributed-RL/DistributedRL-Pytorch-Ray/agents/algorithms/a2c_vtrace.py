from agents.algorithms.base import ActorCritic
from utils.utils import convert_to_tensor, make_mini_batch
from utils.run_env import run_env
import ray
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class A2CVtrace(ActorCritic):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(A2CVtrace, self).__init__(state_dim, action_dim, args)
        self.device = device
        self.args = args

    def get_trajectories(self):
        data = self.data.sample(shuffle=False)
        return data

    def compute_gradients(self, data):
        states, actions, rewards, next_states, dones, old_log_probs = convert_to_tensor(self.device, data['state'],
                                                                                        data['action'], data['reward'],
                                                                                        data['next_state'],
                                                                                        data['done'], data['log_prob'])
        for i in range(self.args['train_epoch']):
            if self.args['discrete']:
                probs = self.get_action(states)
                actions = actions.type(torch.int64)
                log_probs = torch.log(probs.gather(2, actions))
            else:
                mu, std = self.get_action(states)  # dim of state (tra_num, steps, obs_dim)
                dist = Normal(mu, std)
                log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
                entropy = dist.entropy()

            v = self.v(states)
            next_v = self.v(next_states)
            rho, v_trace = self.get_vtrace(rewards, dones, v, next_v, log_probs, old_log_probs)

            critic_loss = F.smooth_l1_loss(v, v_trace[:, :-1])
            actor_loss = - (rho * log_probs * (rewards + self.args['gamma'] *
                                               v_trace[:, 1:] * (1 - dones) - v).detach()).mean()
            entropy_loss = 0  # -entropy.sum(-1).mean()

            loss = actor_loss + critic_loss + 0.01*entropy_loss

            # import time
            # time.sleep(0.03)
            return loss

    def get_vtrace(self, rewards, dones, v, next_v, log_probs, old_log_probs):
        size = rewards.shape[0]
        rho = torch.min(torch.tensor(self.args['rho_bar']), torch.exp(log_probs - old_log_probs))
        c = torch.min(torch.tensor(self.args['c_bar']), torch.exp(log_probs - old_log_probs))
        delta_v = rho * (rewards + self.args['gamma'] * next_v * (1 - dones) - v)
        vtrace = torch.zeros((size, self.args['traj_length'] + 1, 1), device=self.device)

        vtrace[:, -1] = next_v[:, -1]
        vtrace[:, -2] = next_v[:, -1]
        for i in reversed(np.arange(self.args['traj_length']) - 1):
            vtrace[:, i] = v[:, i] + delta_v[:, i] + (1 - dones[:, i]) * \
                           (self.args['gamma'] * c[:, i] * (vtrace[:, i + 1] - next_v[:, i]))

        return rho.detach(), vtrace.detach()
