"""
Model training and validation
"""

import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import logging
try:
	from utils.data import get_dataset
	from utils.dataset_processing.grasp import GraspRectangles, detect_grasps
	from inference.post_process import post_process_output
	import c_grconv_4
	import c_grconv_5
except Exception:
	from CNN.utils.data import get_dataset
	from CNN.utils.dataset_processing.grasp import GraspRectangles
	from CNN.inference.post_process import post_process_output
	import CNN.c_grconv_4
	import CNN.C_grconv_5

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torchsummary import summary

#import cv2
torch.cuda.empty_cache()
global device
global working_dir
working_dir = os.path.abspath(os.path.join(''))
if working_dir not in sys.path:
	sys.path.append(working_dir)

def get_device(use_cpu):
	if torch.cuda.is_available():
		if not use_cpu:
			device = torch.device(device='cuda')
		elif use_cpu:
			device = torch.device(device='cpu')
	else:
		device = torch.device(device='cpu')
	
	return device

def compute_smooth_loss(y, y_pred):
	return F.smooth_l1_loss(y_pred,y)

def compute_mse_loss(y, y_pred):
	return F.mse_loss(y_pred, y)

def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, threshold=0.25, **kwargs):

	grasp_threshold = kwargs.get('grasp_threshold', 0.2)
	grasp_min_distance = kwargs.get('grasp_min_distance', 20)
	
	if not isinstance(ground_truth_bbs, GraspRectangles):
		gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
	else:
		gt_bbs = ground_truth_bbs
	gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps,
		min_distance=grasp_min_distance,
		abs_threshold=grasp_threshold)
	for g in gs:
		if g.max_iou(gt_bbs) > threshold:
			return True
	else:
		return False

def test(model,dataloaders):
	# validate
	model.eval()
	res = {
		'correct':0,
		'fail':0
		}

	print(f"Validating ....")	
	with torch.no_grad():
		for x, y, idx, rot, zoom in dataloaders:
			x = x.to(device)
			y = [yy.to(device) for yy in y]	
			outputs = model(x)
			pos = outputs[0]
			cos = outputs[1]
			sin = outputs[2]
			width = outputs[3]

			q, a, w = post_process_output(pos
				,cos
				,sin
				,width)
			s = calculate_iou_match(q,
								   a,
								   dataloaders.dataset.get_gtbb(idx, rot, zoom),
								   no_grasps=1,
								   grasp_width=w,
								   threshold=0.25,
								   grasp_threshold=0.2,
								   grasp_min_distance=5
								   )

			if s:
				res['correct']+=1
			else:
				res['fail']+=1

	acc = res['correct']/(res['correct']+res['fail'])
	loss = res['fail']/(res['correct']+res['fail'])

	print(f"Accuracy: {res['correct']}/{(res['correct']+res['fail'])} = {'%0.5f'%(acc)}")

	return acc


def train_model(model, optimizer, dataloaders, val_data, num_epochs=5,phase='train',save_dir=None, model_name=None):
	# train
	global working_dir
	model.train()  # Set model to training mode
	best_acc = 0.0
	for epoch in tqdm(range(num_epochs), desc="Epochs"):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		batch_idx = 0
		while batch_idx <= 1000:
			# Iterate over data.
			for inputs, labels , didx, rot, zoom_factor in dataloaders:
				batch_idx += 1
				if batch_idx >= 1000:
					break
				
				inputs = inputs.to(device)
				labels = [yy.to(device) for yy in labels]

				with torch.set_grad_enabled(phase == 'train'):
					
					outputs = model(inputs)
					pos = outputs[0]
					cos = outputs[1]
					sin = outputs[2]
					width = outputs[3]
					r_out = outputs[4]
					# compute loss 
					p_loss = compute_smooth_loss(labels[0],pos)
					c_loss = compute_smooth_loss(labels[1],cos)
					s_loss = compute_smooth_loss(labels[2],sin)
					w_loss = compute_smooth_loss(labels[3],width)
					#r_loss = compute_mse_loss(inputs, r_out)

					loss = p_loss+c_loss+s_loss+w_loss

					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

				epoch_loss = loss / len(dataloaders)
				if batch_idx % 100 == 0:
					print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {'%0.4f'%(epoch_loss)}")
		
		# validate
		acc = test(model, val_data)

		if acc > best_acc or (epoch%10) == 0:
			best_acc = acc
			save_name = save_dir+model_name+'_acc-%0.4f'%(acc)+'_epoch-'+str(epoch)
			torch.save(model, save_name)
			print(f"File writted @ {save_name}")
			print(f"Best accuracy : {best_acc}")
			save_name = ""

	return model


def run(phase=None, d_size=224, **kwargs):
	
	# variable initialization
	global device
	epoch = kwargs.get('epoch',10)
	optimizer_type = kwargs.get('optim_type','adam')
	vis = kwargs.get('vis', False)
	model_name = kwargs.get('m_name', "n_model")
	d_name = kwargs.get('d_name', "cornell")
	dataset_path = kwargs.get('d_path', None)
	save_dir = kwargs.get('save_dir', None)
	m_summary = kwargs.get('summary', False)
	model_type = kwargs.get('m_type', 'res_net')


	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	device = get_device(use_cpu=False)
	print(f"Training on : {device} ....")
	shuffle = True
	size = d_size

	# dataset 
	Dataset = get_dataset(d_name)
	dataset = Dataset(dataset_path,output_size=size,ds_rotate=0.0,
		random_rotate=True,random_zoom=True,
		include_depth=True,include_rgb=True)
		
	print(f"Dataset loaded ....")

	# Creating data indices for training and validation splits
	indices = list(range(dataset.length))
	split = int(np.floor(0.9 * dataset.length))
	if shuffle:
		np.random.seed(123)
		np.random.shuffle(indices)
	
	train_indices, val_indices = indices[:split], indices[split:]
	
	# Creating data samplers and loaders
	train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
	val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
	print(val_sampler)
	train_data = torch.utils.data.DataLoader(
		dataset,
		batch_size=8,
		num_workers=8,
		sampler=train_sampler
	)
	val_data = torch.utils.data.DataLoader(
		dataset,
		batch_size=1,
		num_workers=8,
		sampler=val_sampler
	)
	print(f"Initilization done ....")

	# add network 
	if phase == 'train':

		if model_type == 'res_u_net':
			model = torch.nn.Sequential(c_grconv_5.Res_U_Net())
			print("Training with Res_u-Net -->")
		elif model_type == 'res_net'	:
			model = torch.nn.Sequential(c_grconv_4.ResNet())
			print("Training with ResNet -->")
		model.to(device)
		
		if vis:
			print("Complete model")
			print(model)
			if m_summary:
				summary(model,(4,size,size))
		print(f"Network initialized ....")	

		if optimizer_type == 'adam':
			print(f"Training with Adam optimizer ....")
			optimizer = optim.Adam(model.parameters())
		elif optimizer_type == 'sgd':
			print(f"Training with SGD optimizer ....")
			optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
		elif optimizer_type == 'rmsprop':
			print(f"Training with RMSprop optimizer ....")
			optimizer = optim.RMSprop(model.parameters(), lr = 0.001)
		else:
			optimizer=None
			raise NotImplementedError("optimizer not implemented")

		
		# add training 
		model_ft = train_model(model, optimizer, train_data, val_data,
						   num_epochs=epoch, phase='train', save_dir=save_dir, model_name=model_name)   

		return model_ft

if __name__ == '__main__':
	
	dataset_path = "/home/krish/Downloads/cornell_dataset/"
	d_name = 'cornell'
	
	# GR-ConvNet + custom layers + ResNet
	model_name = 'n_model_res_net'
	model_type = 'res_net'
	save_dir = working_dir+"/generated_model/"+d_name+"/"+"res_net/"
	
	# GR-ConvNet + custom layers + ResNet
	#model_name = 'n_model_res_u_net'
	#model_type = 'res_u_net'
	#save_dir = working_dir+"/generated_model/"+d_name+"/"+"res_u_net/"	

	phase = input("Enter the mode (train): ")

	run(phase=phase, d_size=224, epoch=50, optim_type='rmsprop', vis=True, m_name=model_name, 
		d_name=d_name,d_path=dataset_path,save_dir=save_dir,summary=True, m_type=model_type)
	

