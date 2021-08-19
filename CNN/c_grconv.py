
import torch
import torch.nn as nn
import torch.nn.functional as F

class E_Identity(nn.Module):
	# encoder
	def __init__(self):
		super(E_Identity, self).__init__()
		self.conv1 = nn.Conv2d(128, 64, 4, stride=2, padding=4)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 32, 4, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(32)

		self.conv3 = nn.Conv2d(32, 16, 4, stride=4, padding=1)
		self.bn3 = nn.BatchNorm2d(16)

		self.avg_pooling = nn.AvgPool2d((3,8), stride=4)
		self.flatten = nn.Flatten()

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x_1 = self.bn2(self.conv2(x))
		x = F.relu(self.bn2(self.conv2(x)))

		#x = F.relu(self.bn3(self.conv3(x)))
		x_avg = self.avg_pooling(x_1)
		x_flatten = self.flatten(x_avg)
		return x, x_flatten

class D_Identity(nn.Module):
	# decoder
	def __init__(self):
		super(D_Identity, self).__init__()
		self.conv4 = nn.ConvTranspose2d(16, 32, 4, stride=4, padding=1, output_padding=1)
		self.bn4 = nn.BatchNorm2d(32)

		self.conv5 = nn.ConvTranspose2d(32,64,4,stride=2, padding=1, output_padding=1)
		self.bn5 = nn.BatchNorm2d(64)
		self.conv6= nn.ConvTranspose2d(64,128,4,stride=2, padding=4)
		self.bn6 = nn.BatchNorm2d(128)
	
	def forward(self, x):
		#x = F.relu(self.bn4(self.conv4(x)))
		x = F.relu(self.bn5(self.conv5(x)))
		x = F.relu(self.bn6(self.conv6(x)))

		return x

class Identity2(nn.Module):
	# output layer
	def __init__(self):
		super(Identity2, self).__init__()
		self.pos = nn.Conv2d(32, 1, 2, stride=1)
		self.cos = nn.Conv2d(32, 1, 2, stride=1)
		self.sin = nn.Conv2d(32, 1, 2, stride=1)
		self.width = nn.Conv2d(32, 1, 2, stride=1)

		self.dropout_pos = nn.Dropout(p=0.1)
		self.dropout_cos = nn.Dropout(p=0.1)
		self.dropout_sin = nn.Dropout(p=0.1)
		self.dropout_wid = nn.Dropout(p=0.1)

	def forward(self, x):
		pos_res = self.pos(self.dropout_pos(x))
		cos_res = self.cos(self.dropout_cos(x))
		sin_res = self.sin(self.dropout_sin(x))
		width_res = self.width(self.dropout_wid(x))
		
		return pos_res, cos_res, sin_res, width_res
