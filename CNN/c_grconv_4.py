"""
ResNet 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from grconvnet import *


class encoder(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super(encoder, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
		self.bn = nn.BatchNorm2d(out_channels)

	def forward(self,x, activation=True):
		if activation: x = F.relu(self.bn(self.conv(x)))
		else: x = self.bn(self.conv(x))
		return x

class down_sampling(nn.Module):

	def __init__(self):
		super(down_sampling, self).__init__()
		self.avg_pooling = nn.AvgPool2d((3,8), stride=4)
		self.flatten = nn.Flatten()

	def forward(self,x):
		x_avg = self.avg_pooling(x)
		x_flatten = self.flatten(x_avg)

		return x_flatten


class decoder(nn.Module):
	# decoder
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
		super(decoder, self).__init__()
		
		if output_padding !=None:
			self.conv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size,
				stride=stride, padding=padding,
				output_padding=output_padding)
			self.bn = nn.BatchNorm2d(out_channels)
		else:
			self.conv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size,
				stride=stride, padding=padding)
			self.bn = nn.BatchNorm2d(out_channels)
	
	def forward(self, x, isskip=False, **kwargs):
		skip = kwargs.get('skip')	
		if not isskip:
			x = F.relu(self.bn(self.conv(x)))
		
		if isskip:
			x = torch.cat([x,skip], axis=1)
		
		return x

class Output(nn.Module):
	# output layer
	def __init__(self):
		super(Output, self).__init__()
		self.pos = nn.Conv2d(32, 1, 2, stride=1)
		self.cos = nn.Conv2d(32, 1, 2, stride=1)
		self.sin = nn.Conv2d(32, 1, 2, stride=1)
		self.width = nn.Conv2d(32, 1, 2, stride=1)

		self.conv = nn.Conv2d(32,4,2, stride=1)

		self.dropout_pos = nn.Dropout(p=0.1)
		self.dropout_cos = nn.Dropout(p=0.1)
		self.dropout_sin = nn.Dropout(p=0.1)
		self.dropout_wid = nn.Dropout(p=0.1)

	def forward(self, x):
		pos_res = self.pos(self.dropout_pos(x))
		cos_res = self.cos(self.dropout_cos(x))
		sin_res = self.sin(self.dropout_sin(x))
		width_res = self.width(self.dropout_wid(x))
		out = F.sigmoid(self.conv(x))

		return pos_res, cos_res, sin_res, width_res, out


class ResidualBlock(nn.Module):
	"""
	A residual block with dropout option
	"""

	def __init__(self, in_channels, out_channels, kernel_size=3):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
		self.bn2 = nn.BatchNorm2d(in_channels)

	def forward(self, x_in):
		x = self.bn1(self.conv1(x_in))
		x = F.relu(x)
		x = self.bn2(self.conv2(x))
		return x + x_in


class ResNet(nn.Module):
	def __init__(self):
		super(ResNet,self).__init__()

		self.grconv_encoder = GenerativeResnet_encoder()
		
		self.encoder_1 = encoder(
			in_channels=128, 
			out_channels=64, 
			kernel_size=4, 
			stride=2, 
			padding=4)
		self.encoder_2 = encoder(
			in_channels=64, 
			out_channels=32, 
			kernel_size=4, 
			stride=2, 
			padding=1)
		self.down_sampling = down_sampling()

		self.res1 = ResidualBlock(32, 32)
		self.res2 = ResidualBlock(32, 32)
		self.res3 = ResidualBlock(32, 32)

		self.decoder_1 = decoder(in_channels=32, 
				out_channels=64, 
				kernel_size=4, 
				stride=2, 
				padding=1, 
				output_padding=1)
		self.decoder_2 = decoder(in_channels=64, 
				out_channels=128, 
				kernel_size=4, 
				stride=2, 
				padding=4, 
				output_padding=None)

		self.grconv_decoder = GenerativeResnet_decoder()
		self.output = Output()
		

	def forward(self,x):
		''' grconvnet encoder '''
		e_grconv = self.grconv_encoder(x)

		''' encoder '''
		e_1 = self.encoder_1(e_grconv, activation=True)
		x_1 = self.encoder_2(e_1, activation=False)
		e_2 = self.encoder_2(e_1, activation=True)

		''' down sampling block '''
		ds_x = self.down_sampling(x_1)

		''' res block '''
		res1 = self.res1(e_2)
		res2 = self.res2(res1)
		res3 = self.res3(res2)

		''' decoder '''
		d_1 = self.decoder_1(e_2, isskip=False)
		d_2 = self.decoder_2(d_1, isskip=False)

		''' grconvnet decoder '''
		x = self.grconv_decoder(d_2)

		''' output block '''
		output = self.output(x)

		return output
