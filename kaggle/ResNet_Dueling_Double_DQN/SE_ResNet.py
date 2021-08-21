import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
	def __init__(self, input_channels, num_channels, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=stride, padding=1)
		self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

		self.bn1 = nn.BatchNorm2d(num_channels)
		self.bn2 = nn.BatchNorm2d(num_channels)

		self.shortcut = nn.Sequential()

		if stride != 1 or input_channels != num_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(input_channels, num_channels, padding=0, kernel_size=1, stride=stride),
				nn.BatchNorm2d(num_channels)
			)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = self.relu(out)
		return out

class BottleneckBlock(nn.Module):
	def __init__(self, input_channels, num_channels, stride=1):
		super(BottleneckBlock, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=1)
		self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=1)
		self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=1)

		self.bn1 = nn.BatchNorm2d(num_channels)
		self.bn2 = nn.BatchNorm2d(num_channels)
		self.bn3 = nn.BatchNorm2d(num_channels)

		self.shortcut = nn.Sequential()
		if stride != 1 or input_channels != num_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=stride),
				nn.BatchNorm2d(num_channels)
			)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		return self.relu(out)

class SEBlock(nn.Module):
	def __init__(self, dim, reduction_ratio=4):
		super(SEBlock, self).__init__()
		self.dense = nn.Sequential(
			nn.Linear(dim, dim // reduction_ratio),
			nn.SiLU(inplace=True),
			nn.Linear(dim // reduction_ratio, dim),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = x.mean(axis=(-1, -2))
		y = self.dense(y)
		return x * y.unsqueeze(-1).unsqueeze(-1)

class SEBasicBlock(nn.Module):
	def __init__(self, input_channels, num_channels, stride=1, reduction_ratio=4):
		super(SEBasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=stride, padding=1)
		self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

		self.bn1 = nn.BatchNorm2d(num_channels)
		self.bn2 = nn.BatchNorm2d(num_channels)

		self.shortcut = nn.Sequential()

		if stride != 1 or input_channels != num_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(input_channels, num_channels, padding=0, kernel_size=1, stride=stride),
				nn.BatchNorm2d(num_channels)
			)
		self.relu = nn.ReLU(inplace=True)
		self.se = SEBlock(dim=num_channels, reduction_ratio=reduction_ratio)

	def forward(self, x):
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))

		out = self.se(out)

		out += self.shortcut(x)
		return self.relu(out)

class SEBottleneckBlock(nn.Module):
	def __init__(self, input_channels, num_channels, stride=1, reduction_ratio=4):
		super(SEBottleneckBlock, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=1)
		self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=1)
		self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=1)

		self.bn1 = nn.BatchNorm2d(num_channels)
		self.bn2 = nn.BatchNorm2d(num_channels)
		self.bn3 = nn.BatchNorm2d(num_channels)

		self.shortcut = nn.Sequential()
		if stride != 1 or input_channels != num_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=stride),
				nn.BatchNorm2d(num_channels)
			)
		self.relu = nn.ReLU(inplace=True)
		self.se = SEBlock(dim=num_channels, reduction_ratio=reduction_ratio)

	def forward(self, x):
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))

		out = self.se(out)

		out += self.shortcut(x)
		return self.relu(out)

def ResNet_block(input_channels, num_channels, num_residuals, block, first_block=False):
	blocks = []
	for i in range(num_residuals):
		if i == 0 and not first_block:
			blocks.append(
				block(input_channels=input_channels, num_channels=num_channels, stride=2)
			)  # 第一个block高宽减半
		else:
			blocks.append(
				block(input_channels=num_channels, num_channels=num_channels)
			)
	return blocks

class Encoder(nn.Module):
	def __init__(self, dim_in=12, hidden=32):
		super(Encoder, self).__init__()
		self.gate = nn.Conv2d(dim_in, hidden, 1, padding=(3, 5), padding_mode='circular')
		self.layers = nn.Sequential(
			nn.Sequential(*ResNet_block(hidden, 2 * hidden, num_residuals=2, block=SEBasicBlock)),
			nn.Sequential(*ResNet_block(2 * hidden, 4 * hidden, num_residuals=2, block=SEBasicBlock)),
			nn.Sequential(*ResNet_block(4 * hidden, 8 * hidden, num_residuals=2, block=SEBottleneckBlock)),
			)

	def forward(self, x):
		z = self.gate(x)
		out = self.layers(z)
		return out

if __name__ == '__main__':
	x = torch.rand(1, 12, 7, 11)
	net=Encoder(dim_in=32)
	y=net(x)
	print(y.size())
