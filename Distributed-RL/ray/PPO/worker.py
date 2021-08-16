import gym 
import numpy as np
import torch
import ray
from collections import namedtuple
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'is_terminals'])

@ray.remote
class Worker(object):
    def __init__(self, worker_id, args, shared_storage, replay_buffer, agent):
        self.worker_id = worker_id
        self.args = args
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer
        self.agent = agent
        print("init worker: {}".format(self.worker_id))

    def get_gaes(self, rewards, v_preds, v_preds_next, dones):
        deltas = [r_t + self.args.gamma * (1 - done) * v_next - v for r_t, v_next, v, done in zip(rewards, v_preds_next,
                                                                                             v_preds, dones)]
        advantages = []
        adv = 0.0
        for i in reversed(range(len(deltas))):
            adv = self.args.gae_lambda * self.args.gamma * adv * (1-dones[i]) + deltas[i]
            advantages.append(adv)
        advantages.reverse()
        adv = torch.tensor(advantages, dtype=torch.float32)
        returns = adv + v_preds
        return adv, returns

    def dis_rewards(self, rewards, dones, next_state):
        discounted_reward = np.zeros_like(rewards)
        rollout_length = len(rewards)

        for t in reversed(range(rollout_length)):
            if t == rollout_length - 1:
                discounted_reward[t] = rewards[t] + self.args.gamma * (1 - dones[t]) * self.agent.critic(next_state[-1])
            else:
                discounted_reward[t] = rewards[t] + self.args.gamma * (1 - dones[t]) * discounted_reward[t + 1]
        return torch.tensor(discounted_reward, dtype=torch.float32)

    def preprocess_buffer(self, buffer):
        state = torch.tensor([t.state for t in buffer], dtype=torch.float32)
        next_state = torch.tensor([t.next_state for t in buffer], dtype=torch.float32)
        action = torch.tensor([t.action for t in buffer], dtype=torch.float32).view(-1, self.args.n_a)
        reward = torch.tensor([t.reward for t in buffer], dtype=torch.float32).flatten()
        done = torch.tensor([t.is_terminals for t in buffer], dtype=torch.float32).flatten()
        old_action_log_prob = torch.tensor([t.a_log_prob for t in buffer], dtype=torch.float)

        with torch.no_grad():
            # -----------------------------------------
            # A=r+gamma*v_-v  GAE gae_lambda=0
            # pred_v = self.critic_net(state).flatten()
            # pred_v_ = self.critic_net(next_state).flatten()

            # value_target = reward + self.gamma * pred_v_ * (1 - done)
            # adv = value_target - pred_v
            # Gt = value_target

            if not self.args.use_gae:  # A=Gt-V GAE gae_lambda=1
                Gt = self.dis_rewards(reward, done, next_state)
                adv = Gt - torch.squeeze(self.agent.critic(state))
            else:  # GAE gae_lambda=0.95
                # Gt = self.dis_rewards(reward, done, next_state)
                v = self.agent.critic(state)
                next_v = self.agent.critic(next_state)
                adv, Gt = self.get_gaes(reward, torch.squeeze(v), torch.squeeze(next_v), done)
            # normalize is optional
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            for s, a, returns, adv, a_log in zip(state, action, Gt, adv, old_action_log_prob):
                self.replay_buffer.push.remote([s, a, returns, adv, a_log])

    def run(self):
        print("started worker: {}".format(self.worker_id))
        # build environment
        env = gym.make(self.args.env)
        env.seed(self.worker_id)

        with torch.no_grad():
            while ray.get(self.shared_storage.get_update_counter.remote()) < self.args.max_training_step:
                # get current actor weights
                self.agent.set_weights(ray.get(self.shared_storage.get_weights.remote()))
                self.agent.eval()

                step = 0
                buffer = []
                for i in range(3):
                    state = env.reset()
                    while True:
                        action, action_prob = self.agent.act(state)
                        next_state, reward, done, _ = env.step(action)
                        # add experience to replay buffer
                        trans = Transition(state, action, action_prob, reward/10.0, next_state, done)
                        buffer.append(trans)

                        state = next_state
                        step += 1
                        if done:
                            break

                self.preprocess_buffer(buffer)
                # add executed steps to step counter
                ray.get(self.shared_storage.add_interactions.remote(step))
                ray.get(self.shared_storage.add_sample_counter.remote())
                while ray.get(self.shared_storage.get_sample_counter.remote()) > 0:
                    pass
            env.close()
