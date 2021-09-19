"""
Res-U-Net
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
		self.pos = nn.Conv2d(32, 1, 1, stride=1)
		self.cos = nn.Conv2d(32, 1, 1, stride=1)
		self.sin = nn.Conv2d(32, 1, 1, stride=1)
		self.width = nn.Conv2d(32, 1, 1, stride=1)

		self.conv = nn.Conv2d(32,4,1, stride=1)

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

	def forward(self, x_in, isskip=False, **kwargs):
		skip = kwargs.get('skip')	
		
		if not isskip:
			x = self.bn1(self.conv1(x_in))
			x = F.relu(x)
			x = self.bn2(self.conv2(x))
			
			return x + x_in

		if isskip:
			x = torch.cat([x_in,skip], axis=1)

			return x 


class Res_U_Net(nn.Module):

	def __init__(self):
		super(Res_U_Net, self).__init__()
		
		''' grconvnet - encoder '''
		self.g_conv1 = encoder(in_channels=4, out_channels=32, kernel_size=9, stride=1, padding=4)
		self.g_conv2 = encoder(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
		self.g_conv3 = encoder(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)

		#self.grconv_encoder = GenerativeResnet_encoder()
		self.g_res1 = ResidualBlock(128, 128)
		self.g_res2 = ResidualBlock(128, 128)
		self.g_res3 = ResidualBlock(128, 128)
		self.g_res4 = ResidualBlock(256, 256)
		self.g_res5 = ResidualBlock(384, 384)

		''' custom - encoder '''
		self.c_conv4 = encoder(in_channels=384, out_channels=64, kernel_size=4, stride=2, padding=4)
		self.c_conv5 = encoder(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
		self.down_sampling = down_sampling()

		self.c_res1 = ResidualBlock(32, 32)
		self.c_res2 = ResidualBlock(32, 32)
		self.c_res3 = ResidualBlock(64, 64)

		''' custom - decoder '''
		self.c_t_conv_1 = decoder(in_channels=96, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=1)
		self.c_t_conv_2 = decoder(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=4, output_padding=None)

		''' grconvnet decoder '''
		self.t_conv3 = decoder(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=None)
		self.t_conv4 = decoder(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=1, output_padding=None)
		self.t_conv5 = nn.ConvTranspose2d(64, 32, kernel_size=9, stride=1, padding=4)

		self.output = Output()

		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
				nn.init.xavier_uniform_(m.weight, gain=1)

	def forward(self,x):
		''' grconvnet encoder '''
		e1 = self.g_conv1(x)
		e2 = self.g_conv2(e1)
		e3 = self.g_conv3(e2)

		''' res block '''
		r1 = self.g_res1(e3, isskip=False)
		r2 = self.g_res2(r1, isskip=False)
		r3 = self.g_res3(r2, isskip=False)
		r_s1 = self.g_res4(r3, isskip=True, skip=r2)
		r4 = self.g_res4(r_s1, isskip=False)
		r_s2 = self.g_res5(r4, isskip=True, skip=r1)
		r5 = self.g_res5(r_s2, isskip=False)

		''' encoder '''
		e4 = self.c_conv4(r5, activation=True)
		e5_no_relu = self.c_conv5(e4, activation=False)
		e5 = self.c_conv5(e4, activation=True)

		''' down sampling block '''
		ds_x = self.down_sampling(e5_no_relu)

		''' res block '''
		r6 = self.c_res1(e5, isskip=False)
		r7 = self.c_res2(r6, isskip=False)
		r_s3 = self.c_res3(r7, isskip=True, skip=r6)
		r8 = self.c_res3(r_s3, isskip=False)


		''' decoder '''
		s1 = self.c_t_conv_1(r8, isskip=True, skip=e5)
		d1 = self.c_t_conv_1(s1, isskip=False)
		
		s2 = self.c_t_conv_2(d1, isskip=True, skip=e4)
		d2 = self.c_t_conv_2(s2, isskip=False)

		''' grconvnet decoder '''
		s3 = self.t_conv3(d2, isskip=True, skip=e3)
		d3 = self.t_conv3(s3, isskip=False)
		s4 = self.t_conv4(d3, isskip=True, skip=e2)
		d4 = self.t_conv4(s4, isskip=False)
		s5 = self.t_conv4(d4, isskip=True, skip=e1) # using t_conv4 since only cat
		d5 = self.t_conv5(s5)

		''' output block '''
		output = self.output(d5)
		
		return output
