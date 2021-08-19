import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
try:
	from utils.data import get_dataset
	from utils.dataset_processing.grasp import GraspRectangles, detect_grasps
	from inference.post_process import post_process_output
	from CNN.c_grconv import *
except Exception:
	from CNN.utils.data import get_dataset
	from CNN.utils.dataset_processing.grasp import GraspRectangles
	from CNN.inference.post_process import post_process_output
	from CNN.c_grconv import *

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

class model_preperation(object):

	def __init__(self):
		self.model = None

	def load_pre_trained(self, network_name=None, model_path=None, model_name=None, **kwargs):
		vis = kwargs.get('visual',False)
		m_type = kwargs.get('model_type', 'full')
		if network_name == 'grcnn':
			with open(model_path+'/'+model_name,'br') as m:
				model = torch.load(m)
				m.close()
			if m_type == 'full':
				model = model
			elif m_type == 'encoder':
				model = torch.nn.Sequential(*list(model.children())[:11])
			elif m_type == 'decoder':
				model = torch.nn.Sequential(*list(model.children())[11:16])
			
			if vis==True: print(model)

		return model

def get_device(use_cpu):
	if torch.cuda.is_available():
		if not use_cpu:
			device = torch.device(device='cuda')
		elif use_cpu:
			device = torch.device(device='cpu')
	else:
		device = torch.device(device='cpu')
	
	return device

def compute_loss(y, y_pred):
	return F.smooth_l1_loss(y_pred,y)

def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, threshold=0.25):

	if not isinstance(ground_truth_bbs, GraspRectangles):
		gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
	else:
		gt_bbs = ground_truth_bbs
	gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
	#print(f"gs : {gs}")
	for g in gs:
		#print(f" g max : {g.max_iou(gt_bbs)}")
		if g.max_iou(gt_bbs) > threshold:
			return True
	else:
		return False


def train_model(model, optimizer, dataloaders, num_epochs=5,phase='train'):
	# train
	for epoch in tqdm(range(num_epochs)):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		
		model.train()  # Set model to training mode
		# Iterate over data.
		for inputs, labels , didx, rot, zoom_factor in tqdm(dataloaders):
			inputs = inputs.to(device)
			labels = [yy.to(device) for yy in labels]

			with torch.set_grad_enabled(phase == 'train'):
				
				output = model[:12](inputs)[0]
				outputs = model[12:](output)
				#print(outputs)
				#print(f"len :{len(outputs)}")
				pos = outputs[0]
				cos = outputs[1]
				sin = outputs[2]
				width = outputs[3]
				#exit()
				# compute loss 
				p_loss = compute_loss(labels[0],pos)
				c_loss = compute_loss(labels[1],cos)
				s_loss = compute_loss(labels[2],sin)
				w_loss = compute_loss(labels[3],width)

				loss = p_loss+c_loss+s_loss+w_loss

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		epoch_loss = loss / len(dataloaders)	
		print(f"Epoch {epoch} =  loss : {epoch_loss}")

	return model

def test(model,dataloaders):
	# validate
	model.eval()
	res = {
		'correct':0,
		'fail':0
		}
	with torch.no_grad():
		for x, y, idx, rot, zoom in tqdm(dataloaders):
			x = x.to(device)
			y = [yy.to(device) for yy in y]
			
			output = model[:12](x)[0]
			outputs = model[12:](output)	
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
								   threshold=0.25
								   )

			if s:
				res['correct']+=1
			else:
				res['fail']+=1
			
			#print(res)
	print(f"Acc:{res['correct']/(res['correct']+res['fail'])} \
		Loss : {res['fail']/(res['correct']+res['fail'])}")

	acc = res['correct']/(res['correct']+res['fail'])

	return acc


def run(phase=None, d_size=224, **kwargs):
	global device
	epoch = kwargs.get('epoch',10)
	optimizer_type = kwargs.get('optim_type','adam')
	vis = kwargs.get('vis', False)
	model_name = kwargs.get('m_name', "n_model")
	d_name = kwargs.get('d_name', "cornell")
	dataset_path = kwargs.get('d_path', None)
	save_dir = kwargs.get('save_dir', None)
	m_summary = kwargs.get('summary', False)

	device = get_device(use_cpu=False)
	print(f"Training on : {device}")
	shuffle = True
	size = d_size
	Dataset = get_dataset(d_name)

	dataset = Dataset(dataset_path,output_size=size,ds_rotate=0.0,
		random_rotate=True,random_zoom=True,
		include_depth=True,include_rgb=True)
		
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
		batch_size=4,
		num_workers=8,
		sampler=train_sampler
	)
	val_data = torch.utils.data.DataLoader(
		dataset,
		batch_size=1,
		num_workers=1,
		sampler=val_sampler
	)

	# add network 
	if phase == 'train':
		#model1 = model_preperation().load_pre_trained(network_name='grcnn', 
		#	model_path="trained_model", model_name="model1",model_type='full')
		#print(model1)
		encoder = model_preperation().load_pre_trained(network_name='grcnn', 
			model_path="pretrained_model", model_name="model1",model_type='encoder')
		for params in encoder.parameters():
			params.requires_grad = True # fine tuning 

		decoder = model_preperation().load_pre_trained(network_name='grcnn', 
			model_path="pretrained_model", model_name="model1",model_type='decoder')
		for params in decoder.parameters():
			params.requires_grad = True # fine tuning 

		#torch.autograd.set_detect_anomaly(True)
		model = torch.nn.Sequential(*list(encoder.children()), 
			E_Identity(),
			D_Identity(), 
			*list(decoder.children()),
			Identity2()
			)
		model.to(device)
		
		'''if vis:
			print("Complete model")
			print(model)
			if m_summary:
				summary(model,(4,size,size))
			exit()'''

		if optimizer_type == 'adam':
			optimizer = optim.Adam(model.parameters())
		elif optimizer_type == 'sgd':
			optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
		else:
			optimizer=None
			raise NotImplementedError("optimizer not implemented")

		# Decay LR by a factor of 0.1 every 7 epochs
		#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

		# add training 
		model_ft = train_model(model, optimizer, train_data,
						   num_epochs=epoch, phase='train')
		save_dir = save_dir+model_name
		torch.save(model_ft, save_dir)
		if vis: print(f"File writted @ {save_dir}")    

	if phase == 'val':
		save_dir = save_dir+model_name
		model_ft = torch.load(save_dir)
		if vis: print(f"File loaded from {save_dir}")
		# add validation 
		acc = test(model_ft, val_data)
		return acc

if __name__ == '__main__':
	working_dir = os.path.abspath(os.path.join(''))
	dataset_path = "/home/krish/Downloads/cornell_dataset/"
	d_name = 'cornell'
	model_name = 'n_model_4'
	save_dir = working_dir+"/generated_model/"+d_name+"/"
	phase = input("Enter the mode (train/val): ")

	if working_dir not in sys.path:
		sys.path.append(working_dir)

	if phase == 'train': 
		run(phase=phase, d_size=224, epoch=10, optim_type='adam',
		vis=True, m_name=model_name, d_name=d_name,d_path=dataset_path,save_dir=save_dir,summary=False)
	elif phase == 'val':
		acc_list = []
		for _ in range(5):
			acc = run(phase=phase, d_size=224, m_name=model_name, d_path=dataset_path,save_dir=save_dir)
			acc_list.append(acc)

		print(f"Average accuracy : {sum(acc_list)/len(acc_list)}")
	

