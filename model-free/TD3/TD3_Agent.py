from collections import namedtuple
import os, random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'is_terminals'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, trans):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = trans
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        # Q2
        self.fc4 = nn.Linear(state_dim + action_dim, 400)
        self.fc5 = nn.Linear(400, 300)
        self.fc6 = nn.Linear(300, 1)

    def forward(self, state, action):
        s_a = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(s_a))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2


class TD3(object):
    def __init__(self,
                 env,
                 env_name,
                 buffer_size,
                 gamma,
                 tau,
                 batch_size,
                 policy_noise,
                 noise_clip,
                 policy_freq,
                 lr
                 ):
        self.env_name = env_name
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        self.actor = Actor(self.s_dim, self.a_dim, self.max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(self.s_dim, self.a_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Reply buffer
        self.memory_size = buffer_size
        self.memory = ReplayMemory(self.memory_size)  # s,s_,reward,action
        self.memory_counter = 0

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        if not os.path.exists('./result'):
            os.makedirs('./result')

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.policy_noise = env.action_space.high[0] * policy_noise
        self.noise_clip = env.action_space.high[0] * noise_clip
        self.policy_freq = policy_freq

    def action(self, state):
        state = torch.tensor(state.reshape(1, -1)).float().to(device)
        return self.actor(state).cpu().detach().numpy().flatten()

    def train(self):
        self.num_training += 1

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state = torch.tensor(batch.state).float().view(self.batch_size, -1).to(device)
        action = torch.tensor(batch.action).float().view(self.batch_size, -1).to(device)
        reward = torch.tensor(batch.reward).float().view(self.batch_size, -1).to(device)
        next_state = torch.tensor(batch.next_state).float().view(self.batch_size, -1).to(device)
        done = torch.tensor(batch.is_terminals).float().view(self.batch_size, -1).to(device)

        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
        # Compute target Q-value:
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (1-done) * self.gamma * target_Q
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q.detach()) + F.mse_loss(current_Q2, target_Q.detach())
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates:
        if self.num_training % self.policy_freq == 0:

            # Compute actor losse
            mid, _ = self.critic(state, self.actor(state))
            actor_loss = (-mid).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, transition):
        self.memory.push(transition)
        if self.memory_counter < self.memory_size:
            self.memory_counter += 1

    def save_param(self):
        torch.save(self.actor.state_dict(), './result/' + self.env_name + '_actor_net_params.pkl')
        torch.save(self.critic.state_dict(), './result/' + self.env_name + '_critic_net_params.pkl')

    def load_param(self):
        self.actor.load_state_dict(torch.load('./result/' + self.env_name + '_actor_net_params.pkl'))
        self.critic.load_state_dict(torch.load('./result/' + self.env_name + '_critic_net_params.pkl'))