import torch
from torch import nn

class VGGNet(nn.Module):
	def __init__(self, in_channels=1, num_classes=62, hidden_dim = 16):
		super(VGGNet, self).__init__()
		self.features = nn.Sequential(
			# Block 1
			nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			
			# Block 2
			nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			
			# Block 3
			nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			
			# Block 4
			nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			
			# Block 5
			nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			)
		
		self.fc1 = nn.Sequential(
			nn.Linear(hidden_dim * 8 * 4 * 4, hidden_dim * 8 * 4 * 2),
			nn.ReLU(inplace=True),
			)
			
		self.fc2 = nn.Sequential(
			nn.Linear(hidden_dim * 8 * 4 * 2, hidden_dim * 8 * 4 * 2),
			nn.ReLU(inplace=True),
			)
			
		self.fc_top = nn.Linear(hidden_dim * 8 * 4 * 2, num_classes)
			
	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc_top(x)
		
		return x
