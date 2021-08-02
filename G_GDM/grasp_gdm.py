""" 
Implementation of GDM in custom dataset

"""
import numpy as np 
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
#
import os
import sys
import json
from datetime import datetime
#
from tqdm import tqdm
try:
	#import gtls
	from episodic_gwr import EpisodicGWR
	from preprocessing import processing
except Exception:
	#import gtls
	from .episodic_gwr import EpisodicGWR
	from .preprocessing import processing

# weights and biases
import wandb as w 
import time
from sklearn.preprocessing import normalize

class run_gdm(processing):

	def __init__(self, learning_type, **kwargs):
		
		self.debug = False

		# Network
		self.pre_trained = kwargs.get('pre_trained', False) # bool
		self.g_episodic = kwargs.get('e_net', None)
		self.g_semantic = kwargs.get('s_net', None)
		
		# Data logging
		self.wb_mode = kwargs.get('wb_mode', 'offline')
		self.save_folder = kwargs.get('save_folder', None)
		self.feature_folder = kwargs.get('features_path', "features_256_negative_n_model_7")
		
		# Dataset
		self.batch = kwargs.get('batch', True) # bool
		self.batch_size = kwargs.get('batch_size', 50)
		
		# Training -> hyperparameter selection, 
		parameters = kwargs.get('train_params', None)
		#print(parameters)

		self.dist_type = kwargs.get('dist_type', None)
		self.data_pre_process = kwargs.get('data_pre_process', False)

		# directory handelling
		self.working_dir = os.path.abspath(os.path.join(''))
		if self.working_dir not in sys.path:
			sys.path.append(self.working_dir)

		if not os.path.exists(self.working_dir+'/'+self.save_folder):
			os.makedirs(self.working_dir+'/'+self.save_folder)
		
		# save paramter values
		parameters1 = {}
		parameters1['parameters'] = parameters
		parameters1['distance type'] = self.dist_type
		parameters1['preprocess'] = self.data_pre_process
		parameters1['learning_type'] = learning_type
		parameters1['batch'] = self.batch
		parameters1['batch_size'] = self.batch_size
		
		output_file = open(self.working_dir+'/'+self.save_folder+'/'+"parameters_log.json", "w")
		json.dump(parameters1, output_file, indent=4)
		output_file.close()

		self.learning_type = learning_type # 0-batch, 1-incremental
		
		if self.learning_type == 0: 
			self.l_type = "Batch Learning"
			self.epochs = parameters['epochs'] 
		elif self.learning_type == 1: 
			self.l_type = "Incremental Learning"
			self.epochs = parameters['epochs']

		# training variables 
		self.test_scenes = ['s6','s8', 's14']
		self.total_scenes = 15
		self.scene_folder = os.listdir(self.feature_folder)
		self.e_labels = []
		self.s_labels = [] 

		self.e_label = [len(processing().obj_labels),len(processing().obj_labels)] # instances labels, category labels 	
		self.s_label = [len(processing().obj_labels)] # category labels 
		self.no_classes = self.e_label[0] # total number of intances
		self.no_categories  = self.s_label[0] # total number of categories

		# GDM variables
		self.context = parameters['context']
		self.replay = parameters['memory_replay']
		
		if self.context:
			self.num_context = parameters['num_context']
		else:
			self.num_context = 0

		# hyperparameters
		self.a_threshold = [parameters['g_em_thresh'],parameters['g_sm_thresh']]
		self.beta = parameters['beta']
		self.learning_rates = [parameters['e_b'],parameters['e_n']]
		self.e_regulated = parameters['e_regulated']
		self.s_regulated = parameters['s_regulated']
		self.habn_threshold = parameters['habn_threshold']
		self.rm_threshold = parameters['node_rm_threshold']
		self.max_age = parameters['max_age']

		# log file 
		self.log_file = open(self.working_dir+'/'+self.save_folder+'/'+"log.txt", "w")

		# weights and biases 
		self.wb_init(mode=self.wb_mode)

		self.e_w_logs = ['episodic_activity','episodic_BMU_update_rate','episodic_neuron_update_rate', 'episodic_step_nodes', 
							'episodic_error','episodic_epoch_nodes','episodic_ages','episodic_edges','episodic_epoch_nodes_after_isolated_removal','episodic habituation']
		self.s_w_logs = ['semantic_activity','semantic_BMU_update_rate','semantic_neuron_update_rate', 'semantic_step_nodes', 
							'semantic_error','semantic_epoch_nodes','semantic_ages','semantic_edges','semantic_epoch_nodes_after_isolated_removal', 'semantic habituation']
		self.e_r_w_logs = ['replay_episodic_activity','replay_episodic_BMU_update_rate','replay_episodic__neuron_update_rate', 'replay_episodic_step_nodes', 
							'replay_episodic_error','replay_episodic_epoch_nodes','replay_episodic_ages','replay_episodic_edges',
							'replay_episodic_epoch_nodes_after_isolated_removal', 'replay episodic habituation']
		self.s_r_w_logs = ['replay_semantic_activity','replay_semantic_BMU_update_rate','replay_semantic_neuron_update_rate', 'replay_semantic_step_nodes', 
							'replay_semantic_error','replay_semantic_epoch_nodes','replay_semantic_ages','replay_semantic_edges',
							'replay_semantic_epoch_nodes_after_isolated_removal','replay semantic habituation']									
	def wb_init(self,mode):
		# weights and biases init
		w.login() 
		w.init(project='Grasp dataset', entity='krishkribo',tags=[self.l_type], name="Experiment 1.4", 
			notes="Experimental Test - CNN feature extraction with manhattan distance",mode=mode)
		self.config = w.config
		self.config.learning_rates = self.learning_rates
		self.config.epochs = self.epochs
		self.config.a_threshold = self.a_threshold
		self.config.beta = self.beta
		self.config.replay = self.replay 
		self.config.e_regulated = self.e_regulated
		self.config.s_regulated = self.s_regulated
		self.config.learning_type = self.learning_type
		self.config.Temporal_context = self.context
		self.config.Num_context = self.num_context


	def replay_samples(self, net, size) -> (np.ndarray, np.ndarray):
		samples = np.zeros(size)
		r_weights = np.zeros((net.num_nodes, size, net.dimension))
		r_labels = np.zeros((net.num_nodes, len(net.num_labels), size))
		for i in tqdm(range(0, net.num_nodes)):
			for r in range(0, size):
				if r == 0: samples[r] = i
				else: samples[r] = np.argmax(net.temporal[int(samples[r-1]), :])
				r_weights[i, r] = net.weights[int(samples[r])][0]
				for l in range(0, len(net.num_labels)):
					r_labels[i, l, r] = np.argmax(net.alabels[l][int(samples[r])])
			#print(f"r_weights : {r_weights}")
			#print(f"r_labels : {r_labels}")
			#print(f"len ifi r_weights : {len(r_weights)}")
			#print(f"len ifi r_labels : {len(r_labels)}")
			#exit()
		return r_weights, r_labels
	
	def get_data(self,scene_folder=None, test_scenes=None,batch=False,batch_size=None,train=False,test=False):
		# 
		if self.learning_type == 0: 
			print(f"Getting samples for Batch Learning -->")
			if not batch:
				data = []
				e_labels = []
				s_labels = []
				print(f"Using Full samples -->")
				# complete dataset
				for s in tqdm(scene_folder):
					scene_no = s.split('_')[0]	
					if train: check = scene_no not in test_scenes
					elif test: check = scene_no in test_scenes
					if check:
						# load the data features and labels
						dataset = super().load_file(self.feature_folder+'/'+s)
						data.append([d[0] for d in dataset])
						e_labels.append([d[1] for d in dataset])
						s_labels.append([d[2] for d in dataset])
				
				# combining data from all the scenes into single list 		
				data = np.array([d1 for d2 in data for d1 in d2])
				data = data.reshape((data.shape[0],data.shape[2]))
				e_labels = np.array([d1 for d2 in e_labels for d1 in d2])
				s_labels = np.array([d1 for d2 in s_labels for d1 in d2])

			else:
				print(f"Batch samples of size : {batch_size} -->")
				# trainig_samples based on batch size
				m_dataset = []
				for s in tqdm(scene_folder):
					scene_no = s.split('_')[0]
					if train: check = scene_no not in test_scenes
					elif test: check = scene_no in test_scenes
					if check:
						tmp_dataset = [None]*self.no_classes
						dataset = super().load_file(self.feature_folder+'/'+s)
						# split the dataset by the object instances
						for i in range(self.no_classes):
							# match the elabels to find the classes
							tmp = np.where(np.array([d[1] for d in dataset]) == i)
							# split based on batch size
							tmp_dataset[i] = np.array_split(np.array([dataset[d] for d in tmp[0]]),batch_size)	
						m_dataset.append(tmp_dataset)
				
				m_dataset = np.array(m_dataset, dtype=object)

				# combine all the scenes based on batch sizes  
				if train:
					m_dataset = m_dataset.T
				elif test:
					m_dataset = m_dataset.reshape(m_dataset.shape[2], m_dataset.shape[1], m_dataset.shape[0],
						m_dataset.shape[3], m_dataset.shape[4])

				tmp_dataset2 = [None]*batch_size
				for d1,s in zip(m_dataset,range(batch_size)):
					tmp = []
					for d2 in d1:
						for d3 in d2:
							for d4 in d3:
								tmp.append(d4)
						tmp_dataset2[s] = np.array(tmp)

				m_dataset = np.array(tmp_dataset2,dtype=object)	
				# get the batch data, elabels and slabels
				data = [None]*batch_size
				e_labels = [None]*batch_size
				s_labels = [None]*batch_size

				for d1,s in zip(m_dataset,range(batch_size)):
					#for d2 in d1:
					data[s] = np.array([d2[0] for d2 in d1]) 
					data[s] = data[s].reshape((data[s].shape[0],data[s].shape[2]))
					e_labels[s] = np.array([d2[1] for d2 in d1])
					s_labels[s] = np.array([d2[2] for d2 in d1])

				m_dataset = []
				data = np.array(data,dtype=object)
				e_labels = np.array(e_labels,dtype=object)
				s_labels = np.array(s_labels,dtype=object)				

		elif self.learning_type == 1:
			# incremental learning -> showing different object categories per epoch (example 5 categories 5 epochs)   
			print(f"Getting samples for Incremental Learning -->")
			if not batch:
				print(f"Using Full samples -->")
				m_dataset = []
				# complete dataset
				for s in tqdm(scene_folder):
					scene_no = s.split('_')[0]
					if train: check = scene_no not in test_scenes
					elif test: check = scene_no in test_scenes
					if check:
						tmp_dataset = [None]*self.no_categories
						dataset = super().load_file(self.feature_folder+'/'+s)
						#print(np.unique(np.array([d[2] for d in dataset])))
						for i in range(self.no_categories):
							tmp = np.where(np.array([d[2] for d in dataset]) == i)
							tmp_dataset[i] = np.array([dataset[d] for d in tmp[0]])
						m_dataset.append(tmp_dataset)
				
				tmp_dataset = [] # empty variable to save memory
				m_dataset = np.array(m_dataset,dtype=object)
				if train:
					m_dataset = m_dataset.T
				elif test:
					m_dataset = m_dataset.reshape(m_dataset.shape[1], m_dataset.shape[0],
					 m_dataset.shape[2], m_dataset.shape[3])
					

				def get_array(data,**kwargs):
					data_type = kwargs.get('d_type',None)
					data_type_id = kwargs.get('d_type_id',None)
					tmp = [None]*self.no_categories
					for d1,s1 in zip(data,range(self.no_categories)):
						val=0
						for d2,s2 in zip(d1,range(d1.shape[0])):
							val+= data[s1][s2].shape[0]
						if data_type == 'data':
							tmp[s1] = np.zeros((val, 
								data[s1][0][0][data_type_id].shape[1]))
						else:
							tmp[s1] = np.zeros((val))	

					return np.array(tmp,dtype=object)

				data = get_array(m_dataset, d_type='data', d_type_id=0)
				e_labels = get_array(m_dataset, d_type='e_labels', d_type_id=1)
				s_labels = get_array(m_dataset, d_type='s_labels', d_type_id=2)

				for d1,s1 in zip(m_dataset,range(self.no_categories)):
					count = 0
					for d2,s2 in zip(d1,range(d1.shape[0])):
						for d3 in d2:
							data[s1][count] = d3[0]
							e_labels[s1][count] = d3[1]
							s_labels[s1][count] = d3[2]
							count+=1
				m_dataset = [] # empty variable to save memory
			
			else:
				print(f"Batch samples of size : {batch_size} -->")
				m_dataset = []
				for s in tqdm(scene_folder):
					scene_no = s.split('_')[0]
					if train: check = scene_no not in test_scenes
					elif test: check = scene_no in test_scenes
					if check:
						tmp_dataset = [None]*self.no_categories
						dataset = super().load_file(self.feature_folder+'/'+s)
						#print(np.unique(np.array([d[2] for d in dataset])))
						for i in range(self.no_categories):
							tmp = np.where(np.array([d[2] for d in dataset]) == i)
							tmp_dataset[i] = np.array_split(np.array([dataset[d] for d in tmp[0]]),batch_size)

						#tmp_dataset = np.array(tmp_dataset, dtype=object)
						m_dataset.append(tmp_dataset)

				m_dataset = np.array(m_dataset, dtype=object)
				# empty the variables 
				tmp_dataset = None
				if train:
					m_dataset = m_dataset.T
				elif test:
					m_dataset = m_dataset.reshape(m_dataset.shape[2], m_dataset.shape[1],
					 m_dataset.shape[0], m_dataset.shape[3], m_dataset.shape[4])
				#print(m_dataset.shape)
				tmp_dataset = [None]*batch_size
				for d1,s1 in zip(m_dataset,range(batch_size)):
					#print(d1.shape)
					tmp_dataset1 = [None]*self.no_categories
					for d2,s2 in zip(d1,range(self.no_categories)):
						tmp = []
						#print(d2.shape)
						for d3 in d2:
							for d4 in d3:
								tmp.append(d4)
						tmp = np.array(tmp,dtype=object)
						tmp_dataset1[s2] = tmp

					tmp_dataset[s1] = tmp_dataset1
				m_dataset = np.array(tmp_dataset, dtype=object)
				#print(m_dataset.shape)
				''''if test:
					#m_dataset = m_dataset.T
					print(m_dataset.shape)
					m_dataset = m_dataset.reshape(m_dataset.shape[0], m_dataset.shape[1], m_dataset[0][0].shape[1], m_dataset[0][0].shape[0])
				'''
				"""def get_array(data,**kwargs):
					data_type = kwargs.get('d_type',None)
					data_type_id = kwargs.get('d_type_id',None)
					tmp = np.zeros(shape=(batch_size,self.no_categories))
					tmp = np.array(tmp,dtype=object)
					for d1,s1 in zip(data,range(batch_size)):
						for d2,s2 in zip(d1,range(self.no_categories-1)):
							val=0
							for d3,s3 in zip(d2,range(d2.shape[1])):
								val+=1
							if data_type == 'data':
								tmp[s1][s2] = np.zeros((val,data[s1][0][0][data_type_id].shape[1]))
							else:
								tmp[s1][s2] = np.zeros((val))

					return np.array(tmp,dtype=object)
				print(m_dataset.shape)
				print(m_dataset[0].shape)
				print(m_dataset[0][0].shape)
				print(m_dataset[0][0][0].shape)
				print(m_dataset[0][0][0][0].shape)	
				data = get_array(m_dataset, d_type='data', d_type_id=0)
				e_labels = get_array(m_dataset, d_type='e_labels', d_type_id=1)
				s_labels = get_array(m_dataset, d_type='s_labels', d_type_id=2)
				
				for d1,s1 in zip(m_dataset,range(m_dataset.shape[0])):
					for d2,s2 in zip(d1,range(d1.shape[0])):
						for d3,s3 in zip(d2,range(d2.shape[0])):
							print(d3.shape)
							data[s1][s2][s3] = d3[0]
							e_labels[s1][s2][s3] = d3[1]
							s_labels[s1][s2][s3] = d3[2]

				m_dataset = [] # empty variable to save memory"""
				data = [None]*batch_size
				e_labels = [None]*batch_size
				s_labels = [None]*batch_size

				for d1,s1 in zip(m_dataset,range(batch_size)):
					tmp_dataset1 = [None]*self.no_categories
					tmp_dataset2 = [None]*self.no_categories
					tmp_dataset3 = [None]*self.no_categories	
					for d2,s2 in zip(d1,range(self.no_categories)):
						tmp1 = []
						tmp2 = []
						tmp3 = []
						for d3 in d2:
							tmp1.append(d3[0])
							tmp2.append(d3[1])
							tmp3.append(d3[2])
						tmp1 = np.array(tmp1)
						#print(tmp1.shape)
						tmp1 = tmp1.reshape((tmp1.shape[0],tmp1.shape[2]))
						tmp_dataset1[s2] = np.array(tmp1)
						tmp_dataset2[s2] = np.array(tmp2)
						tmp_dataset3[s2] = np.array(tmp3)

					data[s1] = tmp_dataset1
					e_labels[s1] = tmp_dataset2
					s_labels[s1] = tmp_dataset3	

				data = np.array(data,dtype=object)
				e_labels = np.array(e_labels,dtype=object)
				s_labels = np.array(s_labels,dtype=object)

		return data,e_labels,s_labels
	
	def run(self, f_data, f_e_label, f_s_label, **kwargs):

		save = kwargs.get('save', True)
		#pre_trained = kwargs.get('pre_trained', False)
		#e_net = kwargs.get('e_net', None)
		#s_net = kwargs.get('s_net', None)

		print(f"Training initialized ----->\n")

		# training params
		dist_type = self.dist_type
		data_pre_process = self.data_pre_process
		batch=self.batch
		if batch: batch_size = self.batch_size
		else: batch_size = None

		# updated wandb config
		self.config.update({"batch":batch},allow_val_change=True) 
		self.config.update({"batch_size":batch_size},allow_val_change=True) 

		# G-EM parameters
		g_em_params = {
			"epochs": self.epochs,
			"a_threshold" : self.a_threshold[0],
			"beta": self.beta,
			"l_rates" : self.learning_rates,
			"context": self.context,
			"learning_type": self.learning_type,
			"batch": batch,
			"regulated": self.e_regulated,
			"habn_threshold": self.habn_threshold,
			"max_age": self.max_age,
			"node_rm_threshold": self.rm_threshold
		}

		# G-SM parameters
		g_sm_params = {
			"epochs": self.epochs,
			"a_threshold" : self.a_threshold[1],
			"beta": self.beta,
			"l_rates" : self.learning_rates,
			"context": self.context,
			"learning_type": self.learning_type,
			"batch": batch,
			"regulated": self.s_regulated,
			"habn_threshold": self.habn_threshold,
			"max_age": self.max_age,
			"node_rm_threshold": self.rm_threshold
		}

		print("------------ Train info ------------------")
		print(f_e_label.shape)
		print(f_s_label.shape)
		print(f_e_label[0])
		print(f_s_label[0])
		
		# here episodes are number of the mini batches created (if bacth is true)
		# episode = 1, if batch is false or traning happens in only one of the mini-batch
		if batch:
			#episodes = f_data.shape[0]
			episodes = 1
		else:
			episodes = 1

		init_flag = False
		
		# iterate over episodes
		for episode in tqdm(range(episodes)):
			
			data = f_data
			e_label = f_e_label
			s_label = f_s_label
			
			'''if not batch:
				data = f_data
				e_label = f_e_label
				s_label = f_s_label
			else:
				data = f_data[episode] 
				e_label = f_e_label[episode]
				s_label = f_s_label[episode]'''

			
			# initialize the network only once and update the network over episodes	
			if not init_flag:

				if self.pre_trained != True:
					# episodic memory -- init
					g_episodic = EpisodicGWR()
					g_episodic.init_network(data, self.e_label, random=True, learning_type=self.learning_type, num_context=self.num_context)
					print("\n------ Episodic Network initialized --------\n")
					# semantic memory -- init
					g_semantic = EpisodicGWR()
					g_semantic.init_network(data, self.s_label, random=True, learning_type=self.learning_type, num_context=self.num_context)
					print("\n------ Semantic Network initialized --------\n")
					init_flag = True

				elif self.pre_trained == True:
					
					print("\n------ Episodic Network initialized (pre-trained) --------\n")
					g_episodic = self.g_episodic

					print("\n------ Semantic Network initialized (pre-trained)---------\n")
					g_semantic = self.g_semantic

					init_flag = True

					print(g_episodic.num_nodes)
					print(g_semantic.num_nodes)

			
			if self.learning_type == 0:
				print(f"{self.l_type} -----> started\n")
				d_labels = np.zeros((len(self.e_label),len(e_label)))
				d_labels[0] = e_label
				d_labels[1] = s_label
				#print(data)
				#print(d_labels)
				
				# batch training
				print(f"\n----- Episodic Memory -----\n")

				e_NN, e_qerr, e_acc, e_update_rate_list = g_episodic.train_egwr(
						data, d_labels, g_em_params, w, self.e_w_logs, t_test=True, dist_type=dist_type, data_pre_process=data_pre_process, debug=self.debug) 

				episodic_weights, episodic_labels = g_episodic.test(data,d_labels, ret_vecs=True, dist_type=dist_type, data_pre_process=data_pre_process) 

				print(f"\n----- Semantic Memory -----\n")

				s_NN, s_qerr, s_acc, s_update_rate_list = g_semantic.train_egwr(
						episodic_weights, episodic_labels, g_sm_params, w, self.s_w_logs, t_test=True, dist_type=dist_type, data_pre_process=data_pre_process, debug=self.debug)

				g_episodic.test(data, d_labels, test_accuracy=True, dist_type=dist_type, data_pre_process=data_pre_process)
				w.log({"Accuracy of Episodic Memory":g_episodic.test_accuracy[0]})
				print("Accuracy episodic: %s" %(g_episodic.test_accuracy[0]))

				g_semantic.test(episodic_weights, episodic_labels, test_accuracy=True, dist_type=dist_type, data_pre_process=data_pre_process)
				w.log({"Accuracy of semantic Memory": g_semantic.test_accuracy[0]})		
				print("Accuracy Semantic: %s" %(g_semantic.test_accuracy[0]))

				# updated the error and neuron counters - per episodes 
				episodic_error_counter = e_qerr
				semantic_error_counter = s_qerr
				episodic_neuron_counter = e_NN
				semantic_neuron_counter = s_NN
				episodic_accuracy_counter = e_acc
				semantic_accuracy_counter = s_acc
				episodic_update_rate_counter = e_update_rate_list
				semantic_update_rate_counter = s_update_rate_list

				# log output
				self.log_file.write(f"\n (Training) Learning type : {self.l_type}\n")
				self.log_file.write(f"\nEpoch : {self.epochs}\n")
				self.log_file.write(f"\nAccuracy episodic: {g_episodic.test_accuracy[0]} \n")
				self.log_file.write(f"\nAccuracy semantic: {g_semantic.test_accuracy[0]} \n")
				self.log_file.write(f"\n---------------------------------------------\n")

			elif self.learning_type == 1:
				print(f"{self.l_type} -----> started")
				inc_episodic_error_counter = []
				inc_episodic_neuron_counter = []
				inc_episodic_accuracy_counter = []
				inc_episodic_accuracy_test_counter = []
				inc_episodic_update_counter = []
				inc_semantic_error_counter = []
				inc_semantic_neuron_counter = []
				inc_semantic_accuracy_counter = []
				inc_semantic_accuracy_test_counter = []
				inc_semantic_update_counter = []
				
				# memory replay 
				replay_size = (self.num_context * 2) + 1
				replay_weights = np.array([])
				replay_labels = np.array([])

				# log output
				self.log_file.write(f"\n (Training) Learning type : {self.l_type}\n")
				self.log_file.write(f"\n---------------------------------------------\n")	

				# in incermental learning number of epochs equals the number of categories in the training 
				for epoch in tqdm(range(data.shape[0])):
					print("-----")
					print(e_label[epoch].shape)
					d_labels = np.zeros((len(self.e_label),len(e_label[epoch])))
					d_labels[0] = e_label[epoch]
					d_labels[1] = s_label[epoch]

					#exit()
					self.config.update({"category_epochs":data.shape[0]},allow_val_change=True) 
					
					print(f"----- Episodic Memory -----")

					e_NN, e_qerr, _, e_update_rate_list = g_episodic.train_egwr(
						data[epoch], d_labels, g_em_params, w, self.e_w_logs, t_test=False,
						dist_type=dist_type, data_pre_process=data_pre_process, debug=self.debug) 

					episodic_weights, episodic_labels = g_episodic.test(data[epoch],d_labels, ret_vecs=True, dist_type=dist_type, data_pre_process=data_pre_process) 

					print(f"----- Semantic Memory -----")

					s_NN, s_qerr, _, s_update_rate_list= g_semantic.train_egwr(
						episodic_weights, episodic_labels, g_sm_params, w, self.s_w_logs, t_test=False,
						dist_type=dist_type, data_pre_process=data_pre_process, debug=self.debug)
					
					# replay
					if self.replay == True and epoch > 0:
						g_em_params_cpy = g_em_params
						g_em_params_cpy['context'] = 0
						g_sm_params_cpy = g_sm_params
						g_sm_params_cpy['context'] = 0

						for r in tqdm(range(0, replay_weights.shape[0])):

							print(f"----- Replay Episodic Memory -----")
							g_episodic.train_egwr(
								replay_weights[r], replay_labels[r, :], g_em_params_cpy, w, self.e_r_w_logs, t_test=False,
								dist_type=dist_type, data_pre_process=data_pre_process, debug=self.debug)  
							
							replay_episodic_weights, replay_episodic_labels = g_episodic.test(replay_weights[r], replay_labels[r, :], ret_vecs=True, 
								dist_type=dist_type, data_pre_process=data_pre_process) 

							print(f"----- Replay Semantic Memory -----")

							g_semantic.train_egwr(
									replay_episodic_weights, replay_episodic_labels, g_sm_params_cpy, w, self.s_r_w_logs, t_test=False,
									dist_type=dist_type, data_pre_process=data_pre_process, debug=self.debug)
							
							
					if self.replay:
						#print("replay me")
						replay_weights, replay_labels = self.replay_samples(g_episodic, replay_size)

					# test until the number of categories encountered so far.
					tmp_test_data = data[:epoch+1]
					tmp_test_e_label = e_label[:epoch+1]
					tmp_test_s_label = s_label[:epoch+1]
					tmp_test_data = np.array([val for d in tmp_test_data for val in d])
					tmp_test_e_label = np.array([val for d in tmp_test_e_label for val in d])
					tmp_test_s_label = np.array([val for d in tmp_test_s_label for val in d])
					tmp_test_d_labels = np.zeros((len(self.e_label),len(tmp_test_e_label)))
					tmp_test_d_labels[0] = tmp_test_e_label 
					tmp_test_d_labels[1] = tmp_test_s_label
					#print(test_data.shape)

					test_episodic_weights, test_episodic_labels = g_episodic.test(tmp_test_data, tmp_test_d_labels, test_accuracy=True, ret_vecs=True, dist_type=dist_type, data_pre_process=data_pre_process)
					w.log({"Accuracy of Episodic Memory per episodes":g_episodic.test_accuracy[0]})
					print("Accuracy episodic: %s" %(g_episodic.test_accuracy[0]))
					
					#test_episodic_weights, test_episodic_labels = g_episodic.test(tmp_test_data,tmp_test_d_labels, ret_vecs=True, dist_type=dist_type, data_pre_process=data_pre_process)

					g_semantic.test(test_episodic_weights, test_episodic_labels, test_accuracy=True, dist_type=dist_type, data_pre_process=data_pre_process)
					w.log({"Accuracy of semantic Memory per episode": g_semantic.test_accuracy[0]})		
					print("Accuracy Semantic: %s" %(g_semantic.test_accuracy[0]))

					# updated the error, neuron counters  - epoch 
					'''inc_episodic_error_counter.append(e_qerr)
					inc_semantic_error_counter.append(s_qerr)
					inc_episodic_neuron_counter.append(e_NN)
					inc_semantic_neuron_counter.append(s_NN)
					inc_episodic_update_counter.append(e_update_rate_list)
					inc_semantic_update_counter.append(s_update_rate_list)'''
					
					# updated the error, neuron counters  - category wise
					inc_episodic_error_counter.append(e_qerr[-1])
					inc_semantic_error_counter.append(s_qerr[-1])
					inc_episodic_neuron_counter.append(e_NN[-1])
					inc_semantic_neuron_counter.append(s_NN[-1])
					inc_episodic_update_counter.append(e_update_rate_list[-1])
					inc_semantic_update_counter.append(s_update_rate_list[-1])

					# accuracy
					inc_episodic_accuracy_counter.append(g_episodic.test_accuracy[0])
					inc_semantic_accuracy_counter.append(g_semantic.test_accuracy[0])

					# add the plot network function here to see the network update at the end of every epoch 
					if save:
						plt_val = super().plot_network(g_episodic,edges=True,labels=True,
							title='episodic memory - '+self.l_type,network="episodic", show=False)
						plt_val.savefig(self.working_dir+'/'+self.save_folder+'/'+"e_res_"+str(epoch)+'_'+self.l_type+'_{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
						plt_val.close()

						plt_val = super().plot_network(g_semantic,edges=True,labels=True,
							title='semantic memory - '+self.l_type,network="semantic", show=False)
						plt_val.savefig(self.working_dir+'/'+self.save_folder+'/'+"s_res_"+str(epoch)+'_'+self.l_type+'_{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
						plt_val.close()

					# log output
					self.log_file.write(f"\nEpoch : {epoch}\n")
					self.log_file.write(f"\nAccuracy episodic: {g_episodic.test_accuracy[0]} \n")
					self.log_file.write(f"\nAccuracy semantic: {g_semantic.test_accuracy[0]} \n")
					self.log_file.write(f"\n---------------------------------------------\n")

				# updated the error and neuron counters - epoch 
				'''episodic_error_counter = [j for i in inc_episodic_error_counter for j in i] # combining all the epochs of epoch to epochs e.g 10*3 = 30
				semantic_error_counter = [j for i in inc_semantic_error_counter for j in i]
				episodic_neuron_counter = [j for i in inc_episodic_neuron_counter for j in i]
				semantic_neuron_counter = [j for i in inc_semantic_neuron_counter for j in i]
				episodic_update_rate_counter = j for i in inc_episodic_update_counter for j in i]
				semantic_update_rate_counter = j for i in inc_semantic_update_counter for j in i]'''
				
				# updated the error and neuron counters - category
				episodic_error_counter = inc_episodic_error_counter 
				semantic_error_counter = inc_semantic_error_counter
				episodic_neuron_counter = inc_episodic_neuron_counter
				semantic_neuron_counter = inc_semantic_neuron_counter
				episodic_update_rate_counter = inc_episodic_update_counter
				semantic_update_rate_counter = inc_semantic_update_counter
				# accuracy
				episodic_accuracy_counter = inc_episodic_accuracy_counter
				semantic_accuracy_counter = inc_semantic_accuracy_counter
				
			if save:
				# plot the erros, no of neurons, update rate and accuracies
				super().plot_data(value=episodic_error_counter,title="Episodic Memory - Quantization error - "+str(self.l_type),
					l_type=self.learning_type,y_label="ATQE", path=self.working_dir+'/'+self.save_folder, w=w, plt_type='single', episode=episode, file_name="g-em_qerr")
				super().plot_data(value=semantic_error_counter,title="Semantic Memory - Quantization error - "+str(self.l_type),
					l_type=self.learning_type,y_label="ATQE", path=self.working_dir+'/'+self.save_folder, w=w, plt_type='single', episode=episode, file_name="g-sm_qerr")
				super().plot_data(value=episodic_neuron_counter,title="Episodic Memory - No of Neurons - "+str(self.l_type),
					l_type=self.learning_type,y_label="Neurons", path=self.working_dir+'/'+self.save_folder, w=w, plt_type='single', episode=episode, file_name="g-em_neurons")
				super().plot_data(value=semantic_neuron_counter,title="Semantic Memory - No of Neurons - "+str(self.l_type),
					l_type=self.learning_type,y_label="Neurons", path=self.working_dir+'/'+self.save_folder, w=w, plt_type='single', episode=episode, file_name="g-sm_neurons")
				super().plot_data(value=episodic_update_rate_counter,title="Episodic Memory - Update Rate - "+str(self.l_type),
					l_type=self.learning_type,y_label="Rate", path=self.working_dir+'/'+self.save_folder, w=w, plt_type='single', episode=episode, file_name="g-em_update_rate")
				super().plot_data(value=semantic_update_rate_counter,title="Semantic Memory - Update Rate - "+str(self.l_type),
					l_type=self.learning_type,y_label="Rate", path=self.working_dir+'/'+self.save_folder, w=w, plt_type='single', episode=episode, file_name="g-sm_update_rate")			
				super().plot_data(value=episodic_accuracy_counter,title="Episodic Memory - Accuracy - "+str(self.l_type),
					l_type=self.learning_type,y_label="Accuracy", path=self.working_dir+'/'+self.save_folder, w=w, plt_type='single', episode=episode, file_name="g-em_accuracy")
				super().plot_data(value=semantic_accuracy_counter,title="Semantic Memory - Accuracy - "+str(self.l_type),
					l_type=self.learning_type,y_label="Accuracy", path=self.working_dir+'/'+self.save_folder, w=w, plt_type='single', episode=episode, file_name="g-sm_accuracy")

				# combined graph of episodic and semantic memory
				super().plot_data(value=[episodic_error_counter, semantic_error_counter], title="Quantization error - "+str(self.l_type),
					l_type=self.learning_type,y_label="ATQE", path=self.working_dir+'/'+self.save_folder, w=w, plt_type='multiple', episode=episode, legend=['G-EM', 'G-SM'], file_name="global-qerr")
				super().plot_data(value=[episodic_neuron_counter, semantic_neuron_counter], title="No of Neurons - "+str(self.l_type),
					l_type=self.learning_type,y_label="Neurons", path=self.working_dir+'/'+self.save_folder, w=w, plt_type='multiple', episode=episode, legend=['G-EM', 'G-SM'], file_name="global-neurons")
				super().plot_data(value=[episodic_update_rate_counter, semantic_update_rate_counter], title="Update Rate - "+str(self.l_type),
					l_type=self.learning_type,y_label="Rate", path=self.working_dir+'/'+self.save_folder, w=w, plt_type='multiple', episode=episode, legend=['G-EM', 'G-SM'], file_name="global-update_rate")
				super().plot_data(value=[episodic_accuracy_counter, semantic_accuracy_counter], title="Accuracy - "+str(self.l_type),
					l_type=self.learning_type,y_label="Accuracy", path=self.working_dir+'/'+self.save_folder, w=w, plt_type='multiple', episode=episode, legend=['instance level', 'category level'], file_name="global-train_acc")
				
				# save results 
				super().write_file('episodic_error_counter_'+str(episode)+'_{}'+'.npy'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), episodic_error_counter, path=self.working_dir+'/'+self.save_folder)
				super().write_file('semantic_error_counter_'+str(episode)+'_{}'+'.npy'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), semantic_error_counter, path=self.working_dir+'/'+self.save_folder)
				super().write_file('episodic_neuron_counter_'+str(episode)+'_{}'+'.npy'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), episodic_neuron_counter, path=self.working_dir+'/'+self.save_folder)
				super().write_file('semantic_neuron_counter_'+str(episode)+'_{}'+'.npy'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), semantic_neuron_counter, path=self.working_dir+'/'+self.save_folder)
				super().write_file('episodic_accuracy_counter_'+str(episode)+'_{}'+'.npy'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), episodic_accuracy_counter, path=self.working_dir+'/'+self.save_folder)
				super().write_file('semantic_accuracy_counter_'+str(episode)+'_{}'+'.npy'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), semantic_accuracy_counter, path=self.working_dir+'/'+self.save_folder)
				super().write_file('episodic_update_counter_'+str(episode)+'_{}'+'.npy'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), episodic_update_rate_counter, path=self.working_dir+'/'+self.save_folder)
				super().write_file('semantic_update_counter_'+str(episode)+'_{}'+'.npy'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), semantic_update_rate_counter, path=self.working_dir+'/'+self.save_folder)

				# plotting the network
				plt_val = super().plot_network(g_episodic,edges=True,labels=True,
					title='episodic memory - '+self.l_type,network="episodic", show=True, context=self.context)
				plt_val.savefig(self.working_dir+'/'+self.save_folder+'/'+"e_res_final_"+str(episode)+'_'+self.l_type+'_{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
				plt_val.close()

				plt_val = super().plot_network(g_semantic,edges=True,labels=True,
					title='semantic memory - '+self.l_type,network="semantic", show=True, context=self.context)
				plt_val.savefig(self.working_dir+'/'+self.save_folder+'/'+"s_res_final_"+str(episode)+'_'+self.l_type+'_{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
				plt_val.close()

		return g_episodic, g_semantic


	def test(self,e_net,s_net,test_type='instance', dist_type=None, data_pre_process=False, **kwargs):

		mode = kwargs.get('mode', "both")
		print(f"\nTesting initialized -->")
		
		test_scenes = self.test_scenes

		# for testing batch size is default to False to test on the entire test samples
		batch = False
		batch_size = None

		global_instance_acc = []
		global_category_acc = []
		global_overall_instance_acc = []
		global_overall_category_acc = []
		
		for s in test_scenes:
			instance_acc = []
			category_acc = []
			
			print(f"\nTest scene in process :{s}")
			
			if mode == 'both' or mode == 'categorical':
				print("\nIncremental testing --->")
				self.learning_type = 1
				test_f_data, test_f_e_label, test_f_s_label = self.get_data(scene_folder=self.scene_folder, test_scenes=[s], batch=batch,batch_size=batch_size,train=False,test=True)
				
				if batch:
					# use only first bacth, change index value to test across different bacthes
					idx = 0
					test_f_data = test_f_data[idx]
					test_f_e_label = test_f_e_label[idx]
					test_f_s_label = test_f_s_label[idx]

				# testing based on category wise -> applicable for both batch and incremental testing 
				for epoch in range(test_f_data.shape[0]):
					
					print(f"\nNew instance testing for class : {processing().obj_labels[epoch]} -->")
					
					data = test_f_data[epoch]
					d_labels = np.zeros((len(self.e_label),len(test_f_e_label[epoch])))
					d_labels[0] = test_f_e_label[epoch]
					d_labels[1] = test_f_s_label[epoch]
					
					print(f"\nLabesl in process : {np.array(d_labels[0][0])}, {np.array(d_labels[1][0])}")
					#print(d_labels)

					if test_type == 'instance':
						# instance level testing
						e_net.test(data,d_labels, test_accuracy=True, dist_type=dist_type, data_pre_process=data_pre_process)
						print(f"\ninstance level test accuracy {e_net.test_accuracy[0]}")
						instance_acc.append(e_net.test_accuracy[0])
					
					elif test_type == 'category':
						# category level testing
						episodic_weights, episodic_labels = e_net.test(data,d_labels, test_accuracy=True, ret_vecs=True,
							dist_type=dist_type, data_pre_process=data_pre_process)	
						print(f"\ninstance level test accuracy {e_net.test_accuracy[0]} \n")
						instance_acc.append(e_net.test_accuracy[0])
						s_net.test(episodic_weights,episodic_labels,test_accuracy=True, dist_type=dist_type, data_pre_process=data_pre_process)
						#s_net.test(episodic_weights,d_labels[1].reshape(1,-1),test_accuracy=True, dist_type=dist_type, data_pre_process=data_pre_process)
						print(f"\ncategory level test accuracy {s_net.test_accuracy[0]} \n")
						category_acc.append(s_net.test_accuracy[0])

			if mode == 'both' or mode == 'global':
				print("\nBatch testing --->")
				self.learning_type = 0
				test_f_data, test_f_e_label, test_f_s_label = self.get_data(scene_folder=self.scene_folder,test_scenes=[s],batch=batch,batch_size=batch_size,train=False,test=True)
				
				if batch:
					# use only first bacth, change index value to test across different bacthes
					idx = 0
					test_f_data = test_f_data[idx]
					test_f_e_label = test_f_e_label[idx]
					test_f_s_label = test_f_s_label[idx]

				data = test_f_data
				d_labels = np.zeros((len(self.e_label),len(test_f_e_label)))
				d_labels[0] = test_f_e_label
				d_labels[1] = test_f_s_label
				
				print(data.shape)
				#print(d_labels)
				
				# test on the entire dataset --> with all the classes in one batch for overall accuracy testing 
				# applicable for both bacth and incremental testing 

				if test_type == 'instance':
					# instance level testing
					e_net.test(data,d_labels, test_accuracy=True, dist_type=dist_type, data_pre_process=data_pre_process)
					print(f"\ninstance level test accuracy {e_net.test_accuracy[0]}")
					instance_acc.append(e_net.test_accuracy[0])
				
				elif test_type == 'category':
					# category level testing
					episodic_weights, episodic_labels = e_net.test(data,d_labels, test_accuracy=True, ret_vecs=True,
						dist_type=dist_type, data_pre_process=data_pre_process)	
					print(f"\ninstance level test accuracy {e_net.test_accuracy[0]}")
					instance_acc.append(e_net.test_accuracy[0])
					s_net.test(episodic_weights,episodic_labels,test_accuracy=True, dist_type=dist_type, data_pre_process=data_pre_process)
					#s_net.test(episodic_weights,d_labels[1].reshape(1,-1),test_accuracy=True, dist_type=dist_type, data_pre_process=data_pre_process)
					print(f"\ncategory level test accuracy {s_net.test_accuracy[0]}")
					category_acc.append(s_net.test_accuracy[0])

			if mode == 'both' or mode == 'categorical':
				x_label = [processing().obj_labels[c] for c in range(self.no_categories)]
				#x_label = [processing().obj_labels[c] for c in range(1)]
			
			if mode == 'both' or mode == 'global':
				x_label.append('Avg')
			
			# plot data
			super().plot_data(value=x_label, title=f"Accuracy ({s}) - "+str(self.l_type),
				l_type=None,y_label="Accuracy", path=self.working_dir+'/'+self.save_folder+'/'+s, w=w, plt_type='multiple_bar_both', i_data=instance_acc, c_data=category_acc, show=True, file_name="accuracy_"+s)

			global_instance_acc.append(instance_acc) # remove overall acc in plot
			global_category_acc.append(category_acc) # remove overall acc in plot
			global_overall_instance_acc.append(instance_acc[-1])
			global_overall_category_acc.append(category_acc[-1])

			# log output
			self.log_file.write(f"\n--------- Testing ({s}) -------------\n")
			self.log_file.write(f"\nInstance level accuracy : {instance_acc[-1]}\n")
			self.log_file.write(f"\nCategory level accuracy : {category_acc[-1]}\n")
			self.log_file.write(f"\n-------------------------------------\n")

		# plot data
		super().plot_data(value=x_label, title=f"Accuracy - "+str(self.l_type),
			l_type=None,y_label="Accuracy", path=self.working_dir+'/'+self.save_folder+'/', w=w, plt_type='multiple_bar_category', c_data=global_category_acc, show=True, legend=test_scenes, file_name="global-test_acc")

		# compute overall test accuracy from all classes in all the test scenes
		print(f"----------------------")
		print(f"Instance level avergae accuracy : {np.mean(global_overall_instance_acc)}")
		print(f"Category level avergae accuracy : {np.mean(global_overall_category_acc)}")
		print(f"----------------------")

		# log output
		self.log_file.write(f"\n--------- Testing (overall) -------------\n")
		self.log_file.write(f"\nInstance level avergae accuracy : {np.mean(global_overall_instance_acc)}\n")
		self.log_file.write(f"\nCategory level avergae accuracy : {np.mean(global_overall_category_acc)}\n")
		self.log_file.write(f"\n-----------------------------------------\n")

