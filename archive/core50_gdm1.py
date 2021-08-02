""" 
Implementation of DM-RSOM in CORe50 dataset

"""
import numpy as np 
import matplotlib.pyplot as plt

#
import os
import sys
import multiprocessing
from joblib import Parallel, delayed
#
from tqdm import tqdm
import gtls
from episodic_gwr1 import EpisodicGWR
from preprocessing import processing

# weights and biases
import wandb as w 
import time


class run_gdm(processing):

	def __init__(self):
		# preprocesssing variables 
		self.test_scenes = ['s3','s7','s10']
		self.feature_folder = "features"
		self.data = []
		self.e_labels = []
		self.s_labels = []

		# GDM variables
		print(f"Number of cpu cores available : {multiprocessing.cpu_count()}")
		self.num_cpu = 4
		self.num_context = 1 
		self.e_label = [50,10]
		self.s_label = [10]
		self.no_classes = self.e_label[0]
		self.no_categories  = self.s_label[0]
		self.epochs = 10 # epochs per episode 
		self.learning_type = 0 # 0-batch, 1-incremental

		# hyperparameters
		self.a_threshold = [0.3,0.001]
		self.beta = 0.7
		self.learning_rates = [0.5,0.005]
		self.context = True

		# weights and biases 
		self.wb_init()


	def wb_init(self):
		# weights and biases init
		w.login() 
		w.init(project='GDM_peregrine', entity='krishkribo')
		self.config = w.config
		self.config.learning_rates = self.learning_rates
		self.config.epochs = self.epochs
		self.config.a_threshold = self.a_threshold
		self.config.beta = self.beta


	def get_data(self,batch=False,batch_size=None):

		if self.learning_type == 0: 
			if not batch:
				# complete dataset
				for s in tqdm(os.listdir(self.feature_folder)):
					scene_no = s.split('_')[0]	
					if scene_no not in self.test_scenes:
						# load the data features and labels
						dataset = super().load_file(self.feature_folder+'/'+s)
						self.data.append([d[0] for d in dataset])
						self.e_labels.append([d[1] for d in dataset])
						self.s_labels.append([d[2] for d in dataset])
				
				# combining data from all the scenes into single list 		
				data = np.array([d1 for d2 in self.data for d1 in d2])
				data = data.reshape((data.shape[0],data.shape[2]))
				e_labels = np.array([d1 for d2 in self.e_labels for d1 in d2])
				s_labels = np.array([d1 for d2 in self.s_labels for d1 in d2])

			else:
				# trainig_samples based on batch size
				m_dataset = []
				for s in tqdm(os.listdir(self.feature_folder)):
					scene_no = s.split('_')[0]
					if scene_no not in self.test_scenes:
						tmp_dataset = [None]*self.no_classes
						dataset = super().load_file(self.feature_folder+'/'+s)
						# split the dataset by the object instances
						for i in range(self.no_classes):
							# match the elabels to find the classes
							tmp = np.where(np.array([d[1] for d in dataset]) == i)
							# split based on batch size
							tmp_dataset[i] = np.array_split(np.array([dataset[d] for d in tmp[0]]),batch_size)	
						
						# combine the splited data back based on the classes
						tmp_dataset = np.array(tmp_dataset,dtype=object)
						"""print(tmp_dataset.shape)
						print(tmp_dataset[0].shape)
						print(tmp_dataset[0][0].shape)
						print(tmp_dataset[0][0][0].shape)
						print(tmp_dataset[0][0][-1][1])
						print(tmp_dataset[0][-1][-1][1])
						print("-------------------------")"""
						#tmp_dataset = tmp_dataset.reshape(tmp_dataset.shape[1],tmp_dataset.shape[0])
						tmp_dataset = np.transpose(tmp_dataset)
						"""print(tmp_dataset.shape)
						print(tmp_dataset[0].shape)
						print(tmp_dataset[0][0].shape)
						print(tmp_dataset[0][0][0].shape)
						print(tmp_dataset[0][0][-1][1])
						print(tmp_dataset[0][-1][-1][1])"""

						tmp_dataset1 = [None]*batch_size
						for d1,s in zip(tmp_dataset,range(batch_size)):
							tmp = []
							for d2 in d1:
								for d3 in d2:
									tmp.append(d3)
							tmp_dataset1[s] = np.array(tmp)

						#tmp_dataset = np.array(tmp_dataset1)		
						m_dataset.append(np.array(tmp_dataset1,dtype=object))

				m_dataset = np.array(m_dataset)
				"""print(m_dataset.shape)
				print(m_dataset[0].shape)
				print(m_dataset[0][0].shape)
				print("-----------")"""
				# combine all the scenes based on batch sizes  
				#m_dataset = m_dataset.reshape((m_dataset.shape[1],m_dataset.shape[0]))
				m_dataset = np.transpose(m_dataset)
				
				"""print(m_dataset.shape)
				print(m_dataset[0].shape)
				print(m_dataset[0][0].shape)"""

				tmp_dataset2 = [None]*batch_size
				for d1,s in zip(m_dataset,range(batch_size)):
					tmp = []
					for d2 in d1:
						for d3 in d2:
							tmp.append(d3)
					tmp_dataset2[s] = np.array(tmp)

				m_dataset = np.array(tmp_dataset2,dtype=object)	
				#print(m_dataset.shape)
				#print(m_dataset[0].shape)

				# get the batch data, elables and slabels
				data = [None]*batch_size
				e_labels = [None]*batch_size
				s_labels = [None]*batch_size

				for d1,s in zip(m_dataset,range(batch_size)):
					#for d2 in d1:
					data[s] = np.array([d2[0] for d2 in d1]) 
					data[s] = data[s].reshape((data[s].shape[0],data[s].shape[2]))
					e_labels[s] = np.array([d2[1] for d2 in d1])
					s_labels[s] = np.array([d2[2] for d2 in d1])

				data = np.array(data,dtype=object)
				e_labels = np.array(e_labels,dtype=object)
				s_labels = np.array(s_labels,dtype=object)				

				#print(e_labels[10])

		elif self.learning_type == 1:
			# incremental learning -> showing different object categories per epoch (example 5 categories 5 epochs)   
			m_dataset = []
			if not batch:
				# complete dataset
				for s in tqdm(os.listdir(self.feature_folder)):
					scene_no = s.split('_')[0]
					if not scene_no in self.test_scenes:
						tmp_dataset = [None]*self.no_categories
						dataset = super().load_file(self.feature_folder+'/'+s)
						#print(np.unique(np.array([d[2] for d in dataset])))
						for i in range(self.no_categories):
							tmp = np.where(np.array([d[2] for d in dataset]) == i)
							tmp_dataset[i] = np.array([dataset[d] for d in tmp[0]])
						m_dataset.append(tmp_dataset)
				
				tmp_dataset = []
				m_dataset = np.array(m_dataset,dtype=object)
				m_dataset = m_dataset.reshape((m_dataset.shape[1],m_dataset.shape[0]))
				
				data = [None]*self.no_categories
				e_labels = [None]*self.no_categories
				s_labels = [None]*self.no_categories

				for d1,s in zip(m_dataset,range(self.no_categories)):
					tmp = []
					for d2 in d1:
						for d3 in d2:
							tmp.append(d3)	
					tmp = np.array(tmp)
					data[s] = np.array([d[0] for d in tmp],dtype=object)
					data[s] = data[s].reshape((data[s].shape[0],data[s].shape[2],data[s].shape[1]))
					e_labels[s] = np.array([d[1] for d in tmp],dtype=object)
					s_labels[s] = np.array([d[2] for d in tmp],dtype=object)

				# clear unused vars 
				tmp = []
				m_dataset = []

				data = np.array(data,dtype=object)
				e_labels = np.array(e_labels,dtype=object)
				s_labels = np.array(s_labels,dtype=object)	

			else:
				# based on batch size
				m_dataset = []
				# group by object categories
				for s in tqdm(os.listdir(self.feature_folder)):
					scene_no = s.split('_')[0]
					if scene_no not in self.test_scenes:
						tmp_dataset = [None]*self.no_categories
						dataset = super().load_file(self.feature_folder+'/'+s)
						for i in range(self.no_categories):
							# group by object categories
							tmp = np.where(np.array([d[2] for d in dataset]) == i)
							tmp_dataset[i] = np.array([dataset[d] for d in tmp[0]])
							tmp_dataset[i] = np.array_split(tmp_dataset[i],batch_size)	
						
						tmp_dataset = np.array(tmp_dataset,dtype=object)	
						m_dataset.append(tmp_dataset)
					
				m_dataset = np.array(m_dataset,dtype=object)
				m_dataset = m_dataset.reshape((m_dataset.shape[2],m_dataset.shape[1],m_dataset.shape[0]))
				
				# combine all the sence and create format (batch_size, no_categories) - example (5,10)
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

					tmp_dataset[s1] = np.array(tmp_dataset1,dtype=object)

				tmp_dataset = np.array(tmp_dataset,dtype=object)

				# seperate the data, elabels and slabels of format (batch_size, no_categories, data/elabels/slabels)
				data = [None]*batch_size
				e_labels = [None]*batch_size
				s_labels = [None]*batch_size

				for d1,s1 in zip(tmp_dataset,range(batch_size)):
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
						tmp1 = tmp1.reshape((tmp1.shape[0],tmp1.shape[2],tmp1.shape[1]))
						tmp_dataset1[s2] = np.array(tmp1)
						tmp_dataset2[s2] = np.array(tmp2)
						tmp_dataset3[s2] = np.array(tmp3)

					data[s1] = np.array(tmp_dataset1,dtype=object)
					e_labels[s1] = np.array(tmp_dataset2,dtype=object)
					s_labels[s1] = np.array(tmp_dataset3,dtype=object)	

				data = np.array(data)
				e_labels = np.array(e_labels)
				s_labels = np.array(s_labels)

		#print(data[0].shape)
		#print(e_labels.shape)
		#print(s_labels.shape)
		
		return data,e_labels,s_labels

	def plot_data(self,value,title,x_label,y_label):
		print(value)
		print(len(value))

		data = []
		for i in range(len(value)):
			data.append(value[i])

		for i in range(len(value)):
			plt.plot(data[i])
		
		legend = ["episodes "+str(i) for i in range(len(value))]
		plt.legend(legend,loc='upper right')
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.title(title)

		# weights and biases log
		w.log({title:plt})
	
	def run(self):
		batch=False

		f_data, f_e_label, f_s_label = self.get_data(batch=batch,batch_size=20)
		if batch:
			episodes = f_data.shape[0]
		else:
			episodes = 1

		init_flag = False
		
		episodic_error_counter = []
		episodic_neuron_counter = []
		semantic_error_counter = []
		semantic_neuron_counter = []

		for episode in tqdm(range(episodes)):
			if not batch:
				data = f_data
				e_label = f_e_label
				s_label = f_s_label
			else:
				data = f_data[episode] 
				e_label = f_e_label[episode]
				s_label = f_s_label[episode]

			# initialize the network only once and update the network over episodes	
			if not init_flag:
				# episodic memory -- init
				g_episodic = EpisodicGWR()
				g_episodic.init_network(data, self.e_label, num_context=self.num_context)
				print("------ Episodic Network initialized --------")
				# semantic memory -- init
				g_semantic = EpisodicGWR()
				g_semantic.init_network(data, self.s_label, num_context=self.num_context)
				print("------ Semantic Network initialized --------")
				init_flag = True

			d_lables = np.zeros((len(self.e_label),len(e_label)))
			d_lables[0] = e_label
			d_lables[1] = s_label
			
			if self.learning_type == 0:
				# batch traning
				print(f"----- Episodic Memory -----")
				w_logs = ['episodic_activity','episodic_b_rate','episodic_n_rate', 'episodic_step_nodes', 
					'episodic_error','episodic_epoch_nodes','episodic_ages','episodic_edges','episodic_epoch_nodes_after_isolated_removal']

				e_NN, e_NN2, e_qerr = g_episodic.train_egwr(
					data, d_lables, self.epochs, self.a_threshold[0],
					self.beta, self.learning_rates, self.context, self.num_cpu, 
					self.learning_type, batch, w, w_logs, regulated=1) 

				#print(f"Nodes before isolation removal : {e_NN}")
				#print(f"Nodes afer isolation removal : {e_NN2}")

				episodic_weights, episodic_labels = g_episodic.test(data,d_lables, ret_vecs=True) 
				#print(episodic_labels)
				#time.sleep(10)
				
				print(f"----- Semantic Memory -----")
				w_logs = ['semantic_activity','semantic_b_rate','semantic_n_rate', 'semantic_step_nodes', 
					'semantic_error','semantic_epoch_nodes','semantic_ages','semantic_edges','semantic_epoch_nodes_after_isolated_removal']

				s_NN, s_NN2, s_qerr = g_semantic.train_egwr(
						episodic_weights, episodic_labels, self.epochs,self.a_threshold[1], 
						self.beta, self.learning_rates, self.context, self.num_cpu,
						self.learning_type, batch, w, w_logs, regulated=1)
				
				#print(f"Nodes before isolation removal : {s_NN}")
				#print(f"Nodes afer isolation removal : {s_NN2}")

				# updated the error and neuron counters 
				episodic_error_counter.append(e_qerr)
				semantic_error_counter.append(s_qerr)
				episodic_neuron_counter.append(e_NN)
				semantic_neuron_counter.append(s_NN)

			# log average error of the episode and number of neurons at the end of each episode
			e_qerr_avg = np.sum(e_qerr)/len(e_qerr)
			s_qerr_avg = np.sum(s_qerr)/len(s_qerr)
			w.log({'Average episodic memory ATQE (x-episode)':e_qerr_avg, 'Average semantic memory ATQE (x-episode)':s_qerr_avg})
			w.log({'No of episodic neurons at the end of episode':e_NN2, 'No of semantic neurons at the end of episode':s_NN2})
			# test accurayc per episode
			g_episodic.test(data, d_lables, test_accuracy=True)
			g_semantic.test(episodic_weights, episodic_labels, test_accuracy=True)
			w.log({"Accuracy of Episodic Memory per episodes":g_episodic.test_accuracy[0],
				"Accuracy of semantic Memory per episode": g_semantic.test_accuracy[0]})	

		# plot the erros and no of neurons
		self.plot_data(value=episodic_error_counter,title="Episodic Memory error counter",x_label="episodes",y_label="ATQE")
		self.plot_data(value=semantic_error_counter,title="Semantic Memory error counter",x_label="episodes",y_label="ATQE")
		self.plot_data(value=episodic_neuron_counter,title="Episodic Memory Nodes counter",x_label="episodes",y_label="Nodes")
		self.plot_data(value=semantic_neuron_counter,title="Semantic Memory Nodes counter",x_label="episodes",y_label="Nodes")

		# test accuracy on the entire data	
		f_d_labels = np.zeros((len(self.e_label),len(f_e_label)))
		f_d_labels[0] = f_e_label
		f_d_labels[1] = f_s_label	

		episodic_weights, episodic_labels = g_episodic.test(f_data,f_d_labels, ret_vecs=True)
		g_episodic.test(f_data, f_d_labels, test_accuracy=True) 
		g_semantic.test(episodic_weights, episodic_labels, test_accuracy=True)

		print("Accuracy episodic: %s, semantic: %s" %(g_episodic.test_accuracy[0], g_semantic.test_accuracy[0]))
		w.log({"Accuracy of Episodic Memory on Full dataset":g_episodic.test_accuracy[0],
			"Accuracy of Semantic Memory on Full dataset": g_semantic.test_accuracy[0]})	



if __name__ == '__main__':
	processing = run_gdm()
	
	processing.run()
