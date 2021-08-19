"""
Simulated Robot testing :

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
- perform grasping
	-- pick and place scenario 
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
from time import asctime
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
		self.env = Environment(self.camera, vis=True, debug=debug)
		self.generator = GraspGenerator(self.model_path, self.camera, 5)
		self.obj_list = processing().obj_labels
		self.threshold = 100

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
			rgb, depth = self.get_bounding_box(rgb, depth)
			rgb = rgb[0]
			depth = depth[0]
			_, _, rgb_d = self.generator.predict_grasp(rgb, depth, n_grasps=3)
			
			feature = self.get_features(rgb_d)
			new_object_features.append(feature.reshape(1,-1))

			time.sleep(0.5)
			self.env.remove_all_obj()
		
		return new_object_features

	def get_bounding_box(self, rgb, depth):
		img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
		img1 = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, binary = cv2.threshold(gray, 100, 255, 
		  cv2.THRESH_OTSU)
		inverted_binary = ~binary
		contours, hierarchy = cv2.findContours(inverted_binary,
		  cv2.RETR_TREE,
		  cv2.CHAIN_APPROX_SIMPLE)

		with_contours = cv2.drawContours(img, contours, -1,(255,0,255),3)

		obj_cropped = []
		depth_cropped = []
		for c in contours:
			x, y, w, h = cv2.boundingRect(c)
			# Make sure contour area is large enough
			if (cv2.contourArea(c)) > self.threshold:

				cv2.rectangle(with_contours,(x,y), (x+w,y+h), (255,0,0), 5)
				t_rgb = cv2.resize(img1[y:y+h, x:x+w], (224,224))
				t_depth = cv2.resize(depth[y:y+h, x:x+w], (224,224))

				obj_cropped.append(t_rgb)
				depth_cropped.append(t_depth)

		return obj_cropped, depth_cropped

	def re_train(self, e_net, s_net, object_name, features):
		start_time = int(asctime().split(' ')[-2].split(':')[-1])+60*int(asctime().split(' ')[-2].split(':')[-2])
		
		# expand global categort list 
		self.obj_list.append(object_name)
		print(f"Object categories : {self.obj_list}")
		
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
		
		e_alabels = np.array(e_net.alabels)
		s_alabels = np.array(s_net.alabels)

		e_net.num_labels = [len(self.obj_list), len(self.obj_list)]
		s_net.num_labels = [len(self.obj_list)]

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

		# train 
		process.pre_trained = True
		process.g_episodic = e_net
		process.g_semantic = s_net

		e_net, s_net = process.run(features.reshape((1, features.shape[0], features.shape[1], features.shape[2])), 
			e_label.reshape((1, e_label.shape[0])), 
			s_label.reshape((1, s_label.shape[0])),
			save=False
			)

		end_time = int(asctime().split(' ')[-2].split(':')[-1])+60*int(asctime().split(' ')[-2].split(':')[-2])
		time_diff = end_time-start_time
		print("\nTime in minutes for incremental training of %s object: %2f \n"%(object_name,time_diff/60))

		return e_net, s_net

	def isolated_scenario(self, runs, vis):
		objects = YcbObjects('Grasping/objects/ycb_objects',
								mod_orn=['ChipsCan', 'MustardBottle',
								'TomatoSoupCan'],
								mod_stiffness=['Strawberry'])
		objects.shuffle_objects()
		num_itr = 0
		num_success = 0
		for _ in range(runs):
			for obj_name in objects.obj_names:
				
				# uncomment to load specific object by its name
				'''if obj_name != "PowerDrill":
					continue'''

				if obj_name != "Strawberry" and obj_name !='TennisBall':
					self.threshold = 50
				else: 
					self.threshold = 100

				print(f"Object in process : {obj_name}")
				path, mod_orn, mod_stiffness = objects.get_obj_info(obj_name)
				self.env.load_isolated_obj(path, mod_orn, mod_stiffness)
				self.env.move_away_arm()

				rgb, depth = self.get_rgbd_from_camera()

				rgb, depth = self.get_bounding_box(rgb, depth)
				rgb = rgb[0]
				depth = depth[0]
				grasps, _, rgb_d = self.generator.predict_grasp(rgb, depth, n_grasps=3)

				# get features for gdm learning
				feature = self.get_features(rgb_d)		
				img = rgb
				# predict - GDM
				e_net, s_net, instance_label, category_label = self.predict_obj(self.e_net, self.s_net, feature)

				read_user = input("Train (y/n) : ")
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

				cv2.putText(img, category_label, (0,40), cv2.FONT_HERSHEY_SIMPLEX, 
					1, (0,0,255), 1, cv2.LINE_AA)
				
				grasps, _, rgb_d = self.generator.predict_grasp(rgb, depth, n_grasps=3)

				for i, grasp in enumerate(grasps):
					x, y, z, roll, opening_len, obj_height = grasp
					if vis:
						debug_id = p.addUserDebugLine(
							[x, y, z], [x, y, 1.2], [0, 0, 1], lineWidth=3)

					success_grasp, success_target = self.env.grasp(
						(x, y, z), roll, opening_len, obj_height)
					
					if vis:
						p.removeUserDebugItem(debug_id)
					
					if success_grasp:
						num_success+=1
					if success_target:
						break
					
					self.env.reset_all_obj()
				
				cv2.imshow('Object Recognition', img)
				cv2.imwrite('results/objects_pred_{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+'.jpg', img)
				cv2.imwrite('results/objects_rgb_{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+'.jpg', rgb)
				cv2.imwrite('results/objects_depth_{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+'.jpg', depth)
				#cv2.imwrite('results/objects_rgbd_{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+'.jpg', rgb_image)
				num_itr+=1
				time.sleep(2)
				self.env.remove_all_obj()

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			
			print(f"success/attemps : {num_success}/{num_itr}  = {num_success/num_itr}")
		
		return self.e_net, self.s_net


	def pack_scenario(self,e_net, s_net, runs, vis):

		objects = YcbObjects('Grasping/objects/ycb_objects',
							 mod_orn=['ChipsCan', 'MustardBottle',
									  'TomatoSoupCan'],
							 mod_stiffness=['Strawberry'])
		num_itr = 0
		num_success = 0
		for i in range(runs):
			print(f'Trial {i}')
			straight_fails = 0
			objects.shuffle_objects()
			info = objects.get_n_first_obj_info(5)
			self.env.create_packed(info)
			obj_to_pack = input("Object to pack : ")
			straight_fails = 0
			while len(self.env.obj_ids) != 0 and straight_fails < 3:
				self.env.move_away_arm()
				self.env.reset_all_obj()
				c_rgb, c_depth = self.get_rgbd_from_camera()

				# find the object boundaries using contour detection
				img = cv2.cvtColor(c_rgb, cv2.COLOR_BGR2RGB)
				img1 = cv2.cvtColor(c_rgb, cv2.COLOR_BGR2RGB)

				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				ret, binary = cv2.threshold(gray, 100, 255, 
				  cv2.THRESH_OTSU)
				inverted_binary = ~binary
				contours, hierarchy = cv2.findContours(inverted_binary,
				  cv2.RETR_TREE,
				  cv2.CHAIN_APPROX_SIMPLE)

				with_contours = cv2.drawContours(img, contours, -1,(255,0,255),3)

				cat_objects_found = []
				inst_objects_found = []
				x_arr = []
				y_arr = []
				w_arr = []
				h_arr = []
				# Object Classification
				for c in contours:
					x, y, w, h = cv2.boundingRect(c)
					# Make sure contour area is large enough
					if (cv2.contourArea(c)) > 100:

						cv2.rectangle(with_contours,(x,y), (x+w,y+h), (255,0,0), 5)
						rgb = cv2.resize(img1[y:y+h, x:x+w], (224,224))
						depth = cv2.resize( c_depth[y:y+h, x:x+w], (224,224))
						
						_, _, rgb_d = self.generator.predict_grasp(rgb, depth, n_grasps=1)

						# get features for gdm learning
						feature = self.get_features(rgb_d)
					
						cv2.imshow('image', rgb)

						# predict - GDM
						e_net, s_net, instance_label, category_label = self.predict_obj(e_net, s_net, feature)

						if cv2.waitKey(1) & 0xFF == ord('q'):
							break

						cat_objects_found.append(category_label)
						inst_objects_found.append(instance_label)
						x_arr.append(x)
						y_arr.append(y)
						w_arr.append(w)
						h_arr.append(h)

					cv2.imshow('cropped image', img[y:y+h, x:x+w])
						 
				cv2.imshow('All contours with bounding box', with_contours)
				cv2.destroyAllWindows()

				# Grasping 
				grasps, _, _ = self.generator.predict(c_rgb, c_depth, n_grasps=1)
				grasps_obj, _, _ = self.generator.predict_grasp(c_rgb, c_depth, n_grasps=1)

				for g, grasp in zip(grasps,grasps_obj):
					tmp_y = np.array(y_arr) + np.array(h_arr) - g.center[1]
					tmp_x = np.array(x_arr) + np.array(w_arr) - g.center[0]
					min_idx = np.argmin(tmp_x+tmp_y)
					print(f"object to grasp : {inst_objects_found[min_idx]}, {cat_objects_found[min_idx]}")
					x, y, z, roll, opening_len, obj_height = grasp

					if cat_objects_found[-1] == obj_to_pack or inst_objects_found[-1] == obj_to_pack:
						print("\nPlacing in right (to robot) basket\n")
						# place in basket right to the robot
						succes_grasp, succes_target = self.env.grasp_2(
							(x, y, z), roll, opening_len, obj_height)
					else:
						print("\nPlacing in left (to robot) basket\n")
						# place in basket left to the robot
						success_grasp, succes_target = self.env.grasp(
							(x, y, z), roll, opening_len, obj_height)
					if vis:
						debug_id = p.addUserDebugLine(
							[x, y, z], [x, y, 1.2], [0, 0, 1], lineWidth=3)
						p.removeUserDebugItem(debug_id)
					
					if success_grasp:
						num_success+=1

					if succes_target:
						straight_fails = 0
						break
					else:
						straight_fails += 1

					if straight_fails == 3 or len(self.env.obj_ids) == 0:
						break
				num_itr+=1
			print(f"success/attemps : {num_success}/{num_itr}  = {num_success/num_itr}")


	def run(self, scenario, runs, vis=False, **kwargs):

		e_net = kwargs.get('e_net', None)
		s_net = kwargs.get('s_net', None)

		if scenario == 'isolated':
			e_net, s_net = self.isolated_scenario(runs, vis)
			processing().export_network('e_net_updated',e_net)
			processing().export_network('s_net_updated',s_net)
			processing().write_file("object_list.npy",self.obj_list,"gdm_out/")
			return e_net, s_net

		elif scenario == 'pack':
			e_net = processing().import_network('e_net_updated',EpisodicGWR)
			s_net = processing().import_network('s_net_updated',EpisodicGWR)
			self.obj_list = processing().load_file("gdm_out/object_list.npy")
			# perform manipulation
			self.pack_scenario(e_net, s_net, runs=1, vis=vis)
			
			
if __name__ == "__main__":

	# - use isolated object scenario incase new object is shown to the learning process
	# - for pick and place, and pack scenarios, the object recognition model used needs to have knowledge about all the objects 
	#	in the environment else train it using isolated object scenario and perform manipulation

	arg = argparse.ArgumentParser()
	arg.add_argument("--e_network_path", type=str, default=None, help="Episodic network path (incremental)")
	arg.add_argument("--s_network_path", type=str, default=None, help="Semantic  network path (incremental)")
	arg.add_argument("--model_path", type=str, default=None, help="Model path of custom GRConvNet")
	#arg.add_argument("--params_path", type=str, default=None, help="Parameters path")
	arg.add_argument("--scenario", type=str, default=None, help="isolated/pack")

	args = arg.parse_args()

	# parameters of incremental learning.
	parameters = {}
	parameters['epochs'] = 3
	parameters['context'] = 1
	parameters['num_context'] = 2
	parameters['memory_replay'] = 1
	parameters['g_em_thresh'] = 0.5
	parameters['g_sm_thresh'] = 0.7
	parameters['beta'] = 0.4
	parameters['e_b'] = 0.5
	parameters['e_n'] = 0.005
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
		# to learn new object instances
		e_net, s_net = sim.run(args.scenario, runs=1, vis=False)

	elif args.scenario == "pack":
		# for pick and place, and pack scenarios
		sim.run(args.scenario, runs=1, vis=False)		
	
	