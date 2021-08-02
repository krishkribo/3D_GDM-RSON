"""

Objection Recognition + Grasping 

- load the object to the pybullet
- Get the rgb, depth image from pybullet camera
- Get the rgb-d image from rgb and depth
- load GDM incremental learning networks
- extract 256 dim features using custom GR-ConvNet
- predict the category labels for the input
- if objects is new 
	-- collect samples of the object in the pybullet world
	-- re-train the network to learn new category
	-- predict the category label
- draw the prediction on to the pybullet frame
- perform grasping
	-- isolated objects 
	-- pack scenario

"""

from G_GDM.preprocessing import processing
from G_GDM.grasp_gdm import run_gdm
from G_GDM.episodic_gwr import EpisodicGWR

from Grasping.grasp_generator import GraspGenerator
from Grasping.environment.utilities import Camera
from Grasping.environment.env import Environment
from Grasping.utils import YcbObjects
from Grasping.network.utils.data.camera_data import CameraData
from Grasping.network.utils.dataset_processing.grasp import detect_grasps


import pybullet as p
import argparse
import numpy as np
import os
import time
from datetime import datetime
import sys
import cv2
import json
import torch
import open3d as o3d
from sklearn.preprocessing import normalize



class run_simulation(object):
	def __init__(self, e_net_path=None, s_net_path=None, model_path=None, params=None, debug=False):
		
		# directory handelling
		self.model_path = model_path
		self.params = params
		self.e_net = e_net_path
		self.s_net = s_net_path

		self.e_net = processing().import_network(self.e_net, EpisodicGWR)
		self.s_net = processing().import_network(self.s_net, EpisodicGWR)

		# environment
		center_x, center_y = 0.05, -0.52
		self.camera = Camera((center_x, center_y, 1.9), (center_x,
						center_y, 0.785), 0.2, 2.0, (224, 224), 40)
		self.env = Environment(self.camera, vis=False, debug=debug)
		self.generator = GraspGenerator(self.model_path, self.camera, 5)
		self.obj_list = processing().obj_labels

	def get_rgbd_from_camera(self):
		# get image from camera 
		rgb, depth, _ = self.camera.get_cam_img()

		return rgb, depth

	def get_features(self, rgbd_img, model_name='n_model_7'):
		# get gdm features
		return processing().get_feature(data=rgbd_img, norm=True, model_name=model_name)

	def predict_obj(self, e_net, s_net, data):

		data = data.reshape(1,-1)
		
		e_weight, e_label = e_net.test(np.array(data), None, ret_vecs=True, data_pre_process=False,
			dist_type = 'manhattan')
		s_label = s_net.test(e_weight, e_label, data_pre_process=False,
			dist_type = 'manhattan')

		print(f"\nPredicted instance label : {self.obj_list[int(e_label[0])]}\n")
		print(f"\nPredicted category label : {self.obj_list[int(s_label[0])]}\n")
		
		return e_net, s_net, e_label, self.obj_list[int(s_label[0])]


	def collect_samples(self, path, mod_orn, mod_stiffness):
		new_object_features = []
		self.env.remove_all_obj()
		for _ in range(30):
			self.env.load_isolated_obj(path, mod_orn, mod_stiffness)
			rgb, depth = self.get_rgbd_from_camera()
			_, _, rgb_d = self.generator.predict_grasp(rgb, depth, n_grasps=3)
			
			feature = self.get_features(rgb_d)
			new_object_features.append(feature.reshape(1,-1))

			time.sleep(0.5)
			self.env.remove_all_obj()
		
		return new_object_features

	def re_train(self, e_net, s_net, object_name, features):
		
		# expand global categort list 
		#if object_name not in processing().obj_labels:
		
		self.obj_list.append(object_name)
		
		print(self.obj_list)
		# get e_label and s_label
		e_label = []
		s_label = []

		obj_idx = self.obj_list.index(object_name)
		for i in range(len(features)):
			# replace in same instance of the same category object is retrained 
			e_label.append(len(self.obj_list)-1)
			# get the index of the new added object
			s_label.append(len(self.obj_list)-1)

		features = np.array(features)
		e_label = np.array(e_label)
		s_label = np.array(s_label)
		
		# print(features.shape)
		# print(e_label.shape)
		# print(s_label.shape)

		# init the gdm learning
		process = run_gdm( 
						learning_type=1,
						wb_mode="offline", 
						save_folder="results",
						features_path="G_GDM/features_256_negative_n_model_7",
						batch=False,
						batch_size=None,
						train_params=self.params,
						dist_type="manhattan",
						data_pre_process=False
						)

		# print(np.array(e_net.alabels).shape)
		# print(np.array(s_net.alabels).shape)
		
		e_alabels = np.array(e_net.alabels)
		s_alabels = np.array(s_net.alabels)

		e_net.num_labels = [len(self.obj_list), len(self.obj_list)]
		s_net.num_labels = [len(self.obj_list)]

		# e_net.alabels = -np.ones((e_alabels.shape[0], e_alabels.shape[1]+1,
		# 	len(self.obj_list)), dtype=object)
		# s_net.alabels = -np.ones((s_alabels.shape[0], s_alabels.shape[1]+1,
		# 	len(self.obj_list)), dtype=object)

		e_net.alabels = []
		for l in range(0, len(e_net.num_labels)):
			e_net.alabels.append(-np.ones((e_net.num_nodes, e_net.num_labels[l])))

		for l in range(e_alabels.shape[0]):
			for s in range(e_alabels.shape[1]):
				for c in range(e_alabels.shape[2]):
					e_net.alabels[l][s][c] = e_alabels[l][s][c]
		
		s_net.alabels = []
		for l in range(0, len(s_net.num_labels)):
			s_net.alabels.append(-np.ones((s_net.num_nodes, s_net.num_labels[l])))					

		for l in range(s_alabels.shape[0]):
			for s in range(s_alabels.shape[1]):
				for c in range(s_alabels.shape[2]):
					s_net.alabels[l][s][c] = s_alabels[l][s][c]

		# print(np.array(e_net.alabels).shape)
		# print(np.array(s_net.alabels).shape)

		# train 
		process.pre_trained = True
		process.g_episodic = e_net
		process.g_semantic = s_net

		e_net, s_net = process.run(features.reshape((1, features.shape[0], features.shape[1], features.shape[2])), 
			e_label.reshape((1, e_label.shape[0])), 
			s_label.reshape((1, s_label.shape[0])),
			save=False
			)


		return e_net, s_net

	def isolated_scenario(self, runs, vis):
		objects = YcbObjects('Grasping/objects/ycb_objects',
								mod_orn=['ChipsCan', 'MustardBottle',
								'TomatoSoupCan'],
								mod_stiffness=['Strawberry'])
		objects.shuffle_objects()

		for _ in range(runs):
			for obj_name in objects.obj_names:
				print(f"Object in process : {obj_name}")
				path, mod_orn, mod_stiffness = objects.get_obj_info(obj_name)
				self.env.load_isolated_obj(path, mod_orn, mod_stiffness)
				self.env.move_away_arm()

				rgb, depth = self.get_rgbd_from_camera()

				grasps, _, rgb_d = self.generator.predict_grasp(rgb, depth, n_grasps=3)

				# get features for gdm learning
				feature = self.get_features(rgb_d)
				
				img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

				# predict - GDM
				e_net, s_net, instance_label, category_label = self.predict_obj(self.e_net, self.s_net, feature)

				read_user = input("Re-train (y/n) : ")
				if read_user == 'y' or read_user == 'yes':
					category_object = input("Enter the object name :")

					# collect samples for retraining
					n_features = self.collect_samples(path, mod_orn, mod_stiffness)

					# re-train 
					self.re_train(e_net, s_net, category_object, n_features)

					# predict 
					e_net, s_net, instance_label, category_label = self.predict_obj(self.e_net, self.s_net, feature)
					
					self.e_net = e_net
					self.s_net = s_net

					self.env.load_isolated_obj(path, mod_orn, mod_stiffness)

				cv2.putText(img, category_label, (0,30), cv2.FONT_HERSHEY_SIMPLEX, 
					0.5, (0,0,255), 1, cv2.LINE_AA)

				for i, grasp in enumerate(grasps):
					x, y, z, roll, opening_len, obj_height = grasp
					if vis:
						debug_id = p.addUserDebugLine(
							[x, y, z], [x, y, 1.2], [0, 0, 1], lineWidth=3)

					succes_grasp, succes_target = self.env.grasp(
						(x, y, z), roll, opening_len, obj_height)
					if vis:
						p.removeUserDebugItem(debug_id)
					
					self.env.reset_all_obj()
				
				cv2.imshow('Object Recognition', img)
				cv2.imwrite('results/objects_{}.jpg'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), img)
				time.sleep(2)
				self.env.remove_all_obj()

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			
		return self.e_net, self.s_net

	def pack(self,runs, vis):
		
		raise NotImplementedError()
	
	def pack_selected_scenario(self,runs, vis):
		
		raise NotImplementedError()

	def run(self, scenario, runs, vis=False, **kwargs):

		e_net = kwargs.get('e_net', None)
		s_net = kwargs.get('s_net', None)

		if scenario == 'isolated':
			e_net, s_net = self.isolated_scenario(runs, vis)
			return e_net, s_net

		elif scenario == 'pack':
			self.pack_scenario(runs, vis)
			return None, None

		elif scenario == 'pack_selected':
			self.pack_selected_scenario(runs, vis)
			return None, None
		



if __name__ == "__main__":

	# - use isolated object scenario incase new object is shown to the learning process
	# - for pack scenario with object recognition used network which has knowledge about all the objects 
	#	in the environment else train them isolated object scenario

	arg = argparse.ArgumentParser()
	arg.add_argument("--e_network_path", type=str, default=None, help="Episodic network path (incremental)")
	arg.add_argument("--s_network_path", type=str, default=None, help="Semantic  network path (incremental)")
	arg.add_argument("--model_path", type=str, default=None, help="Model path of custom GRConvNet")
	arg.add_argument("--params_path", type=str, default=None, help="Parameters path")
	arg.add_argument("--scenario", type=str, default=None, help="isolated/pack")

	args = arg.parse_args()
	parameters = {}
	parameters['epochs'] = 3
	parameters['context'] = 1
	parameters['num_context'] = 2
	parameters['memory_replay'] = 1
	parameters['g_em_thresh'] = 0.7
	parameters['g_sm_thresh'] = 0.8
	parameters['beta'] = 0.5
	parameters['e_b'] = 0.3
	parameters['e_n'] = 0.003
	parameters['e_regulated'] = 0
	parameters['s_regulated'] = 1
	parameters['habn_threshold'] = 0.1
	parameters['node_rm_threshold'] = 0.2
	parameters['max_age'] = 2000

	sim = run_simulation(e_net_path=args.e_network_path, s_net_path=args.s_network_path, 
		model_path=args.model_path, 
		params=parameters,
		debug=False)
	
	if args.scenario == "isolated":
		e_net, s_net = sim.run(args.scenario, runs=1, vis=False)
	
	elif args.scenario == "pack_selected":
		# check the obj in isolated 
		#e_net, s_net = sim.run('isolated', runs=1, vis=False)
		# run pack selected
		sim.run(args.scenario, runs=1, vis=False)
