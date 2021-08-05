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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=500):
        super(MLP, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))      # activation function for hidden layer
        x = torch.tanh(self.hidden2(x))
        x = self.predict(x)             # linear output
        return x


class NNDynamicsModel(nn.Module):
    def __init__(self,
                 env,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 env_name="HalfCheetah-v3"
                 ):
        super(NNDynamicsModel, self).__init__()
        self.batch_size = batch_size
        self.iter = iterations
        self.env_name = env_name
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]

        self.mean_s, self.std_s, self.mean_deltas, self.std_deltas, self.mean_a, self.std_a = normalization

        self.net = MLP(n_input=self.s_dim+self.a_dim, n_output=self.s_dim).to(device)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        if not os.path.exists('./result'):
            os.makedirs('./result')

    def fit(self, data):
        normalization = compute_normalization(data)
        self.mean_s, self.std_s, self.mean_deltas, self.std_deltas, self.mean_a, self.std_a = normalization

        s = np.concatenate([d["state"] for d in data])  # (timestep, 8)
        sp = np.concatenate([d["next_state"] for d in data])
        a = np.concatenate([d["action"] for d in data])  # (timestep, 2)

        # normalize
        s_norm = (s - self.mean_s) / (self.std_s + 1e-7)
        a_norm = (a - self.mean_a) / (self.std_a + 1e-7)

        s_a = np.concatenate((s_norm, a_norm), axis=1)  # 网络输入
        deltas_norm = ((sp - s) - self.mean_deltas) / (self.std_deltas + 1e-7)  # 网络输出

        # train
        torch_dataset = Data.TensorDataset(torch.tensor(s_a,dtype=torch.float32), torch.tensor(deltas_norm,dtype=torch.float32))
        loader = Data.DataLoader(
            dataset=torch_dataset,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,  # random shuffle for training
            num_workers=2,  # subprocesses for loading data
        )
        for epoch in tqdm.tqdm(range(self.iter)):  # train entire dataset 3 times
            for step, (b_x, b_y) in enumerate(loader):  # for each training step
                batch_x = b_x.to(device)
                batch_y = b_y.to(device)

                output = self.net(batch_x)
                loss = self.loss_func(output, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states
        and (unnormalized) actions and return the (unnormalized) next states
        as predicted by using the model """
        """ YOUR CODE HERE """
        #normalize
        s_norm = (states - self.mean_s) / (self.std_s + 1e-7)
        a_norm = (actions - self.mean_a) / (self.std_a + 1e-7)
        s_a = np.concatenate((s_norm, a_norm), axis=1)

        delta = self.net(torch.tensor(s_a, dtype=torch.float32).to(device)).cpu().detach().numpy()

        #denormalize
        return delta * self.std_deltas + self.mean_deltas + states

    def save_params(self):
        torch.save(self.net.state_dict(), os.path.join('./result/',self.env_name+'_dynamics_net_params.pkl'))

    def load_params(self):
        self.net.load_state_dict(torch.load(os.path.join('./result/',self.env_name+'_dynamics_net_params.pkl')))



