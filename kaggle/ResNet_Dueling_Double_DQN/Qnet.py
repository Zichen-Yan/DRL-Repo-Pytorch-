import torch
from torch import nn
import torch.nn.functional as F
from SE_ResNet import *

class Qnet(nn.Module):
	def __init__(self, dim_in=12, dim_out=3, hidden=32):
		super(Qnet, self).__init__()
		self.encoder = Encoder(dim_in, hidden=hidden)
		self.dense = nn.Sequential(
			nn.Conv2d(8*hidden, 8*hidden, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(8*hidden),
			nn.Conv2d(8*hidden, 128, 1),
			nn.ReLU(inplace=True)
		)
		self.advantage = nn.Sequential(
			nn.Linear(128, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, dim_out)
		)
		self.value=nn.Sequential(
			nn.Linear(128, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 1)
		)

	def forward(self, x):
		state = self.encoder(x)
		p = self.dense(state)
		p = p.mean(axis=(-1, -2))

		adv = self.advantage(p)
		v = self.value(p)
		avgadv = torch.mean(adv, dim=1, keepdim=True)
		return v+adv-avgadv, v

if __name__ == '__main__':
	x = torch.rand(1, 12, 7, 11)
	net = Qnet(dim_in=32,dim_out=3)
	y = net(x)
	print(y)
