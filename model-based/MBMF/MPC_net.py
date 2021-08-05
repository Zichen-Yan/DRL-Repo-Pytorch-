import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import torch.utils.data as Data
import tqdm
from utils import compute_normalization
import os
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=64):
        super(MLP, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden*2)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden*2, n_hidden*4)
        self.hidden3 = torch.nn.Linear(n_hidden * 4, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = torch.relu(self.hidden1(x))      # activation function for hidden layer
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = torch.tanh(self.predict(x))           # linear output
        return x


class MPC_net(nn.Module):
    def __init__(self,
                 env,
                 env_name,
                 normalization,
                 batch_size=500,
                 iterations=300,
                 learning_rate=1e-4
                 ):
        super(MPC_net, self).__init__()
        self.batch_size = batch_size
        self.iter = iterations
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]
        self.env_name = env_name

        self.mean_s, self.std_s, self.mean_deltas, self.std_deltas, self.mean_a, self.std_a = normalization

        self.net = MLP(n_input=self.s_dim, n_output=self.a_dim).to(device)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        if not os.path.exists('./result'):
            os.makedirs('./result')

    def fit(self, s, a):
        # normalization = compute_normalization(data)
        # self.mean_s, self.std_s, self.mean_deltas, self.std_deltas, self.mean_a, self.std_a = normalization
        #
        # s = np.concatenate([d["state"] for d in data])  # (timestep, 8)
        # a = np.concatenate([d["action"] for d in data])  # (timestep, 2)


        # normalize
        self.mean_s = np.mean(s, 0)
        self.std_s = np.std(s, 0)
        s_norm = (s - self.mean_s) / (self.std_s + 1e-7)
        # a_norm = (a - self.mean_a) / (self.std_a + 1e-7)
        a_norm = a

        # train
        torch_dataset = Data.TensorDataset(torch.tensor(s_norm, dtype=torch.float32), torch.tensor(a_norm, dtype=torch.float32))
        loader = Data.DataLoader(
            dataset=torch_dataset,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,  # random shuffle for training
            num_workers=2,  # subprocesses for loading data
        )
        for epoch in tqdm.tqdm(range(self.iter)):
            for step, (b_x, b_y) in enumerate(loader):
                batch_x = b_x.to(device)
                batch_y = b_y.to(device)

                output = self.net(batch_x)
                loss = self.loss_func(output, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def get_action(self, states):
        #normalize
        s_norm = (states - self.mean_s) / (self.std_s + 1e-7)

        mu = self.net(torch.tensor([s_norm], dtype=torch.float32).to(device))
        std = torch.ones_like(mu, dtype=torch.float32)

        dist = Normal(mu, std)
        action = dist.sample()
        act_log_prob = dist.log_prob(action).cpu()

        action = torch.squeeze(action).cpu()
        action = torch.clamp(action, -1, 1)

        return action.tolist(), act_log_prob, torch.squeeze(mu).cpu().detach().tolist()

    def save_params(self):
        torch.save(self.net.state_dict(), os.path.join('./result/', self.env_name + '_mpc_net_params.pkl'))

    def load_params(self):
        self.net.load_state_dict(torch.load(os.path.join('./result/', self.env_name + '_mpc_net_params.pkl')))



