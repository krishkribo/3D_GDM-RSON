 # GGDM end-end pipeline
'''
input -> point cloud data/ point cloud - file source
output -> grasp detection, GDM-RSON learning (incermental) 
'''
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


try:
	from preprocessing import processing
	from grasp_gdm import run_gdm
	from episodic_gwr import EpisodicGWR
	from CNN.utils.dataset_processing import grasp

except Exception:
	from .preprocessing import processing
	from .grasp_gdm import run_gdm
	from .episodic_gwr import EpisodicGWR
	from .. CNN.utils.dataset_processing import grasp

import os 
import sys
from time import time, sleep
from datetime import datetime
import numpy as np
from skimage.feature import peak_local_max
from scipy.signal import find_peaks
import cv2
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt



class run_rt(object):
	def __init__(self, data=None, **kwargs):
		d_flag = kwargs.get('d_flag', False)
		r_flag = kwargs.get('r_flag', False)
		e_net = kwargs.get('e_network',None) 
		s_net = kwargs.get('s_network', None)
		self.n_grasps = kwargs.get('n_grasp', 1)
		self.m_name = kwargs.get('m_name', None) # pre-trained model
		self.d_dist = kwargs.get('min_dist', None)
		self.offset_dist = 30
		self.threshold = 0.25
		# load network
		g_episodic = processing().import_network(e_net, EpisodicGWR)
		g_semantic = processing().import_network(s_net, EpisodicGWR)

		''' GDM testing and learning '''
		if len(data) <=1: 
			data = data[0]
			if os.path.isfile(data): pass
			else: raise Exception("input data is not a file")
			# get data
			d_img, rgb_img = processing().get_data(pcl_file=data, width=224, height=224)
			# show the image 
			if d_flag:
				processing().show_image(data=d_img)
			elif r_flag:
				processing().show_image(data=rgb_img)

			# get features 
			feature = processing().get_feature(data=d_img, norm=True, model_name=self.m_name)
			
			# prediction 
			self.predict(g_episodic, g_semantic, feature)

			''' grasp synthesis '''
			# single samples is singleton set entropy calculation is eliminated
			self.get_grasp(d_img, rgb_img)
		
		else:
			entropy_imgs = []
			for i,d in enumerate(data):
				if os.path.isfile(d): pass
				else: raise Exception("input data is not a file")
				# get data
				d_img, rgb_img = processing().get_data(pcl_file=d, width=224, height=224)
				# show the image 
				if d_flag:
					processing().show_image(data=d_img)
				elif r_flag:
					processing().show_image(data=rgb_img)

				# get features 
				feature = processing().get_feature(data=d_img, norm=True, model_name=self.m_name)
				
				# prediction 
				self.predict(g_episodic, g_semantic, feature)
				
				# get entropy
				entropy_imgs.append(processing().get_entropy_img(img=d_img, gray_scale=True, norm=True))
				
				self.get_grasp(d_img, rgb_img)

			# input with the maximum entropy is used for grasping	
			max_entropy = np.argmax(entropy_imgs)
			max_entropy_data = data[max_entropy]
			d_img, rgb_img = processing().get_data(pcl_file=max_entropy_data, width=224, height=224)
			self.get_grasp(d_img, rgb_img)

	def	predict(self, e_net, s_net, sample):
		sample =  normalize(sample.reshape(1, -1)) 
		e_weights, e_labels = e_net.test(np.array(sample), None, ret_vecs=True, data_pre_process=False,
			dist_type = 'manhattan')
		print(f"Predicted instance label : {e_labels[0][0]}")
		s_label = s_net.test(e_weights, e_labels, data_pre_process=False,
			dist_type = 'manhattan')
		print(f"Predicted category label : {s_label[0][0]}")

	def predict_grasp(self, q_img, angle_img, w_img):
		min_peak = processing().get_minima(img=q_img, offset=self.offset_dist, n_grasp=self.n_grasps, min_dist=self.d_dist, show=False, save=True, thresh=self.threshold)
		#max_peak = peak_local_max(q_img, min_distance=20, threshold_abs=0.25, num_peaks=self.n_grasps)
		#print(min_peak)
		grasps = []
		for grasp_point_array in min_peak:
			grasp_point = tuple(grasp_point_array)

			grasp_angle = angle_img[grasp_point]

			g = grasp.Grasp(grasp_point, grasp_angle)
			if w_img is not None:
				g.length = w_img[grasp_point]
				g.width = g.length / 2

			grasps.append(g)

		return grasps

	def get_grasp(self, d_img, rgb_img):
		# encoder 
		e_out, e_res = processing().model_predict(data=d_img, model_type='c_encoder', model_name=self.m_name)
		# decoder 
		_, q_res = processing().model_predict(data=e_out, model_type='c_decoder', img_type='quality', model_name=self.m_name)
		_, angle_res = processing().model_predict(data=e_out, model_type='c_decoder', img_type='angle', model_name=self.m_name)
		_, width_res = processing().model_predict(data=e_out, model_type='c_decoder', img_type='width', model_name=self.m_name)
		
		#print(q_res.shape)
		#print(angle_res.shape)
		#print(width_res.shape)		

		# estimate the grasp
		grasp_pts = self.predict_grasp(q_res, angle_res, width_res)

		# plot grasp
		#plot_grasp(fig= plt.figure(figsize=(10, 10)), 
		#	rgb_img=rgb_img, grasps=grasp_pts, save=True)
		# plot result
		processing().plot_grasp(plt.figure(figsize=(10,10)), rgb_img, d_img, q_res, angle_res, 
			width_res, grasp_pts, self.offset_dist, plt_type='full', save=True)
		sleep(1)
		processing().plot_grasp(plt.figure(figsize=(10,10)), rgb_img, d_img, q_res, angle_res, 
				width_res, grasp_pts, self.offset_dist, plt_type='depth', save=True)
		sleep(1)
		processing().plot_grasp(plt.figure(figsize=(10,10)), rgb_img, d_img, q_res, angle_res, 
				width_res, grasp_pts, self.offset_dist, plt_type='grasp', save=True)

if __name__ == '__main__':
	
	working_dir = os.path.abspath(os.path.join(''))
	if working_dir not in sys.path:
		sys.path.append(working_dir)

	pcl_folder = r"test_pcl_files"
	file = os.listdir(working_dir+'/'+pcl_folder)
	#print(file)
	
	# 
	# single sample test
	#run_rt(data=[working_dir+'/'+pcl_folder+'/'+file], e_network='e_net_batch', s_network='s_net_batch',
	#	d_flag=True, r_flag=True, n_grasp=1, m_name='n_model_7')
	# multiple sample test 
	run_rt(data=[working_dir+'/'+pcl_folder+'/'+f for f in file], e_network='e_net_batch', 
		s_network='s_net_batch', d_flag=True, r_flag=True, n_grasp=2, m_name='n_model_4', min_dist=5)