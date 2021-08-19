#! /usr/bin/env python3

"""
Main program : Dataset generation - Sequentail 3D - object Dataset

Source : shapenet Dataset

pipeline :
	- connect to ros service and publishers
	- initialize the get ROI (Region of interest)
	- spawn the object model as per the object list
	- set the object model states -> translation (x,y,z) and rotation (roll,picth,yaw)
	- set the data augmentation parameters
	- start the marker and bounding box services
	- extract the point cloud based on the current view
	- save the point cloud (current view) to .pcd format
"""

from utils import *
from utils.add_model import generate_model
from utils.get_pcl import pcl_helper
from utils.draw_markers import draw_marker
from utils.get_ROI import get_object_roi
import rospy
#from gazebo_msgs.msg import ModelState
from bounding_box_server.srv import GetBoundingBox, GetBoundingBoxRequest
from gazebo_msgs.srv import SpawnModel, DeleteModel, DeleteModelResponse
from gazebo_msgs.srv import SetModelState, GetModelState
from sensor_msgs.msg import PointCloud2,PointField
from sensor_msgs import point_cloud2
import open3d
import os
import sys
import numpy as np
import time
from time import asctime
import argparse
import math
from tqdm import tqdm
"""
To start the ROI service 
RUN : python3 utils/object_detection.py in seperate terminal
"""

class data_generation(object):

	def __init__(self):
		# calling class functions 
		self.gen_model = generate_model()
		self.save_pcl = pcl_helper()
		self.get_roi = get_object_roi()
		# add the workspace to the system path
		sys_path = os.path.abspath(os.path.join(".."))
		if sys_path not in sys.path:
			sys.path.append(sys_path)

		# os / system variable declarations 
		self.model_dir = '../models/sim_models/meshes'
		self.dataset_dir = '../dataset/'
		# get the avaliable model names and their scale size
		self.models_dict = {}
		self.scale_dir = []
		self.scene_dir = []
		scene_dir = []
		for m_dir in os.listdir(os.path.abspath(self.model_dir)):
			if os.path.isdir(self.model_dir+'/'+m_dir):
				if m_dir[0] == 's':
					scene_dir.append(m_dir)
					for s_dir in os.listdir(os.path.abspath(self.model_dir+'/'+m_dir+'/')): 
						if s_dir.split('_')[0] == 'scale':
							self.scene_dir.append(m_dir)
							self.scale_dir.append(s_dir)
							self.models_dict[m_dir+'_'+str(float(s_dir.split('_')[-1])/100)] = os.listdir(
								os.path.abspath(self.model_dir+'/'+m_dir+'/'+s_dir))
		#print(self.models_dict)
		# get all the model names 
		self.models = []
		for (_,value),s_data,c_data in zip(self.models_dict.items(),self.scene_dir,self.scale_dir):
			for m in value:
				if not os.path.isdir(self.model_dir+'/'+s_data+'/'+c_data+'/'+m):
					self.models.append(m.split('.')[0])

		# get the unique models to set the model index for data annotation
		self.unique_model = list(np.unique(np.array(self.models)))
		
		# create dataset directory
		if not os.path.exists(self.dataset_dir):
			os.mkdir(self.dataset_dir)

		# spawn model service
		self.spawn_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
		rospy.loginfo("Waiting for spawn service")
		self.spawn_service.wait_for_service()
		rospy.loginfo("Connected to spawn service")
		# delete service
		self.delete_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
		rospy.loginfo("Waiting for delete service")
		self.delete_service.wait_for_service()
		rospy.loginfo("Connected to delete service")
		# set model service 
		self.set_model_service = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
		rospy.loginfo("Waiting for set model service")
		self.set_model_service.wait_for_service()
		rospy.loginfo("Connected to set model service")
		# Get model service
		self.get_model_service = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
		rospy.loginfo("Waiting for get model service")
		self.get_model_service.wait_for_service()
		rospy.loginfo("Connected to get model service")
		# bouding box service 
		self.bounding_box_client = rospy.ServiceProxy("/get_bounding_boxes", GetBoundingBox)
		rospy.loginfo("Waiting for bounding box service")
		self.bounding_box_client.wait_for_service()
		rospy.loginfo("Connected to set bouding box service")
		
		self.pcl_topic = "/point_cloud_functions/cloud"
		# support variables
		self.n_frames = 500
		self.x_start = 0.176
		self.x_end =  -0.7
		self.y_start = 0.072
		self.y_end = -0.2475
		self.binary = False # boolean True-> without RGB pcl, False -> with RGB pcl
		self.add_noise = False
		self.downsample = False
		self.mu = list(np.arange(0.01,0.51,0.01))
		self.sigma = list(np.arange(0.005,0.011,0.001))
		self.ds_percent = list(np.arange(0.1,1.005,0.05))
		self.degree_shift = 30  # shift in angles for r,p,y
		self.roll_range = np.arange(0, 360, self.degree_shift+60)
		self.pitch_range = np.arange(0, 360, self.degree_shift+30)
		self.yaw_range = np.arange(0, 720, self.degree_shift)
		self.total_no_scenes = 15
		self.scene_dir_len = len(scene_dir)
		self.scenes_per_set = int(self.total_no_scenes/self.scene_dir_len)

	def get_bounding_boxes(self):
		box_msg = GetBoundingBoxRequest()
		box_msg.transform_to = "new_frame_id"
		box_msg.cluster_threshold = 0.05
		box_msg.surface_threshold = 0.010
		box_msg.min_x = -2.5
		box_msg.max_x = 2.5

		try:
			self.bounding_box_client(box_msg)
		except Exception as e:
			rospy.loginfo("Error in getting bounding box output")

	def get_model_state(self,model_name):
		try:
			status = self.get_model_service(model_name,'world') # model state in gazebo world
			return status
		except Exception as e:
			print("The error status : ", e)
			return 
		
	def delete_model(self):
		for model in self.models:
			try:
				status = self.get_model_state(model)
				if not status.status_message.split(':')[-1] == " model does not exist":
					# note : need space before model in ' model does not exist check'
					self.gen_model.delete_model_request(model,self.delete_service)
				else: pass
			except Exception as e:
				print("The error status : ",e)
	
	def to_radians(self,degree):
		return (degree*math.pi)/180

	def to_degree(self,radians):
		return int((radians*180)/math.pi)

	def run(self):
		# delete all the models if exist in the environment
		self.delete_model()
		for (scene_scale,model_name),scale_dir,scene_dirs in tqdm(zip(self.models_dict.items(),self.scale_dir,self.scene_dir),desc='scale_scene bar'):
			
			# each objects in scenes are used for generation of 4 scenes 
			for itr in range(self.scenes_per_set):

				# get the scene number
				if itr == 0:
					scene_no = int(scene_scale.split('_')[0][1:])
					#continue
				else:
					scene_no = scene_no + self.scene_dir_len

				scene_dir = 's'+str(scene_no)
				print(f"scene in process : {scene_dir}")
				
				#get the scale value
				scale = float(scene_scale.split('_')[-1])

				# create scene directory
				if not os.path.exists(self.dataset_dir+'/'+scene_dir):
					os.mkdir(self.dataset_dir+'/'+scene_dir)
				
				for model in tqdm(model_name,desc='model bar'):
					#model = 'mug'
					if os.path.isfile(self.model_dir+'/'+scene_dirs+'/'+scale_dir+'/'+model) and model.split('.')[-1] == 'dae':
						model = model.split('.')[0]
						model_index = self.unique_model.index(model)
						
						# uncomment the following lines for generating the dataset of specific object from specific scene
						'''if model == 'hammer' and scene_dir == 's1':
							print(model)
							print(scene_dir)
						else: continue'''

						# create object directory -> dataset
						model_dir = self.dataset_dir+'/'+scene_dir+'/'+model+'/'
						if not os.path.exists(model_dir):
							os.mkdir(model_dir)
						elif os.path.exists(model_dir):
							continue
						# spawn model at the origin
						rospy.loginfo("Spawning model: %s"%model)
						self.spawn_service.call(
							self.gen_model.create_model_request(
								scene_dirs+'/'+scale_dir+'/'+model, scale, 
								0.0, 0.0, 0.003076,  # model location, scale, position
								0.0, 0.0, 0.0 # rotation
							)
						)
						rospy.loginfo("model name :%s spawned"%model)
						rospy.sleep(0.1) # wait time before another request
						
						start_time = int(asctime().split(' ')[-2].split(':')[-1])+60*int(asctime().split(' ')[-2].split(':')[-2])
						self.x_range = np.arange(self.x_start, self.x_end, -0.1)
						frame_count = 0
						# model transitions --> pose
						import random  
						for _ in tqdm(self.x_range, desc='translation bar'):
							x = random.sample(list(self.x_range), 1) # getting unique sample
							x = x[0]
							self.y_range = np.arange(self.y_start, self.y_end, -0.05)
							y = random.sample(list(self.y_range),1)
							y = y[0]
							# model transition --> orientation
							for r in tqdm(self.roll_range,desc='rotation bar -> roll'):
								#print('roll: ',r)
								r= self.to_radians(r)
								for p in tqdm(self.pitch_range,desc='rotation bar -> pitch'):
									#print('pitch: ',p)
									p = self.to_radians(p)
									for yaw in tqdm(self.yaw_range,desc='rotation bar -> yaw'):
										#print('yaw: ',yaw)
										yaw = self.to_radians(yaw)
										self.filename = model_dir + 'C_'+'{:02}'.format(scene_no)+'_'+'{:02}'.format(model_index)+ \
											'_'+'{:03}'.format(self.to_degree(r))+'_'+'{:03}'.format(self.to_degree(p))+\
											'_'+'{:03}'.format(self.to_degree(yaw))+'_'+'{:03}'.format(frame_count)
										# start draw marker module
										# To see visual in rviz use topic -> visual_marker
										draw_marker()
										# bounding box service message
										self.get_bounding_boxes()
										#rospy.loginfo("model name :%s"%model)
										model_state = self.get_model_state(model)
										#rospy.loginfo(model_state.pose)
										self.set_model_service.call(
											self.gen_model.set_model_request(model,
											x, y, 0.003076,
											r, p, yaw, 1.0)
										)
										rospy.sleep(0.1) # rest time
										# save pcl
										msg = rospy.wait_for_message(self.pcl_topic, PointCloud2, timeout=None)
										
										if itr == 0:
											self.add_noise = False
											self.downsample = False
											self.save_pcl.run_pcl(msg, self.filename+'.pcd', self.binary, add_noise=self.add_noise, 
												mu=None, sigma=None, downsample=self.downsample, ds_percent=None)
										
										elif itr > 0:
											mu = None
											sigma = None
											ds = None
											if int((frame_count*100)/self.n_frames) < 50 and int((frame_count*100)/self.n_frames) >= 15:
												self.add_noise = True
												self.downsample = False
												mu = random.sample(self.mu,1)[0]
												sigma = random.sample(self.sigma,1)[0]
												self.save_pcl.run_pcl(msg, self.filename+'.pcd', self.binary, add_noise=self.add_noise, 
													mu=mu, sigma=sigma, downsample=self.downsample, ds_percent=None)
											
											elif int((frame_count*100)/self.n_frames) < 75 and int((frame_count*100)/self.n_frames) >= 50:
												self.downsample = True
												self.add_noise = False
												ds = random.sample(self.ds_percent,1)[0]
												self.save_pcl.run_pcl(msg, self.filename+'.pcd', self.binary, add_noise=self.add_noise, 
													mu=None, sigma=None, downsample=self.downsample, ds_percent=ds)

											elif int((frame_count*100)/self.n_frames) <= 100 and int((frame_count*100)/self.n_frames) >= 75:
												self.add_noise = True
												self.downsample = True
												mu = random.sample(self.mu,1)[0]
												sigma = random.sample(self.sigma,1)[0]
												ds = random.sample(self.ds_percent,1)[0]
												self.save_pcl.run_pcl(msg, self.filename+'.pcd', self.binary, add_noise=self.add_noise, 
													mu=mu, sigma=sigma, downsample=self.downsample, ds_percent=ds)

											else:
												self.add_noise = False
												self.downsample = False
												self.save_pcl.run_pcl(msg, self.filename+'.pcd', self.binary, add_noise=self.add_noise, 
													mu=None, sigma=None, downsample=self.downsample, ds_percent=None)
										
										if frame_count < self.n_frames:
											frame_count+=1
											flag = 0
										if frame_count == self.n_frames:
											rospy.loginfo("All the frames are obtained")
											flag = 1
											#exit() # uncomment to generate dataset of only one scene. 
											break
										time.sleep(0.2) # rest time

									if flag == 1: break
									else: pass
									
								if flag == 1: break
								else: pass  

							if flag == 1: break
							else: pass

						end_time = int(asctime().split(' ')[-2].split(':')[-1])+60*int(asctime().split(' ')[-2].split(':')[-2])
						time_diff = end_time-start_time
						rospy.loginfo("Time in minutes to process model : %s :%2f"%(model, time_diff/60))

						# delete the model  
						self.gen_model.delete_model_request(model,self.delete_service)
						rospy.loginfo("%s deleted"%model)
						time.sleep(0.5) # wait time before another request

		self.get_roi.flag = True
		rospy.signal_shutdown("process done")

if __name__ == "__main__":
	get_args = argparse.ArgumentParser(description="3D object sequence dataset generation")
	args = get_args.parse_args()
	rospy.init_node("workspace_init")
	run = data_generation()
	run.run()