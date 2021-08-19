"""
GDM - Learnig 
-- Batch Learning 
-- Incremental Learning
-- Continual Learning
"""

import argparse
import json

try:
	from grasp_gdm import *
	from preprocessing import processing
except Exception:
	from .grasp_gdm import *
	from .preprocessing import processing

import os
import sys


def run(args, parameters, t):
	l_type = None # learning type
	# variable intialization based on the learning type
	if args.learning_type.lower() == 'batch':

		if int(args.context):
			save_folder = args.output_folder+'/'+'exp_'+str(t)+'_tc'
			e_file = 'e_net_batch_tc'
			s_file = 's_net_batch_tc'
		
		elif not int(args.context):
			save_folder = args.output_folder+'/'+'exp_'+str(t)+'_no_tc'
			e_file = 'e_net_batch_no_tc'
			s_file = 's_net_batch_no_tc'

		l_type = 0

	elif args.learning_type.lower() == 'incremental':
		
		if int(args.memory_replay):
			save_folder = args.output_folder+'/'+'exp_'+str(t)+'_replay'
			e_file = 'e_net_incremental_replay'
			s_file = 's_net_incremental_replay'
		
		elif not int(args.memory_replay):
			save_folder = args.output_folder+'/'+'exp_'+str(t)+'_no_replay'
			e_file = 'e_net_incremental_no_replay'
			s_file = 's_net_incremental_no_replay'

		l_type = 1
		
	else:
		l_type = None
		raise Exception("Enter the correct learning type")

	# intializing  the GDM learning
	if not args.pre_trained:
		process = run_gdm(learning_type=l_type,
							wb_mode=args.wandb_mode, 
							save_folder=save_folder,
							features_path=args.features_path,
							batch=bool(args.mini_batch),
							batch_size=args.bs_size,
							train_params=parameters,
							dist_type=args.dist_type,
							data_pre_process=bool(args.preprocess)
							)
	elif args.pre_trained:
		process = run_gdm(pre_trained=bool(args.pre_trained),
							e_net=args.episodic_network,
							s_net=args.semantic_network,	
							learning_type=l_type,
							wb_mode=args.wandb_mode, 
							save_folder=save_folder,
							features_path=args.features_path,
							batch=bool(args.mini_batch),
							batch_size=args.bs_size,
							train_params=parameters,
							dist_type=args.dist_type,
							data_pre_process=bool(args.preprocess)
							)


	# training
	if args.mode.lower() == 'both' or args.mode.lower() == 'train':
		
		# shuffle the traning set for each episode 
		scene_folder = random.sample(process.scene_folder,len(process.scene_folder))
		
		''' uncomment below lines for debugging'''
		#print(scene_folder)
		#print(f"args.continual_learning_mode : {args.continual_learning_mode}")
		
		f_data, f_e_label, f_s_label = process.get_data(scene_folder=scene_folder,test_scenes=process.test_scenes,batch=bool(args.mini_batch),batch_size=args.bs_size,train=True,test=False)
			
		if args.mini_batch:
			idx = int(args.bs_size/2)
			data = f_data[idx]
			e_label = f_e_label[idx]
			s_label = f_s_label[idx]
		else:
			data = f_data
			e_label = f_e_label
			s_label = f_s_label

		# NI - Incremental Learning with memory replay
		if args.continual_learning_mode == 'NI':
			# use the first index of scene folder which is not in test scenes

			train_data = np.empty((data.shape[0]), dtype=object)
			re_train_data = np.empty((data.shape[0]), dtype=object)
			train_e_label = np.empty((e_label.shape[0]), dtype=object)
			re_train_e_label = np.empty((e_label.shape[0]), dtype=object)
			train_s_label = np.empty((s_label.shape[0]), dtype=object)
			re_train_s_label = np.empty((s_label.shape[0]), dtype=object)

			for d in range(data.shape[0]):
				train_data[d] = data[d][:int(500/args.bs_size)]
				re_train_data[d] = data[d][int(500/args.bs_size):]
				train_e_label[d] = e_label[d][:int(500/args.bs_size)]
				re_train_e_label[d] = e_label[d][int(500/args.bs_size):]
				train_s_label[d] = s_label[d][:int(500/args.bs_size)]
				re_train_s_label[d] = s_label[d][int(500/args.bs_size):]

			# train
			print(f"Training -->")
			#time.sleep(5)
			process.batch = False
			e_net, s_net = process.run(train_data, train_e_label, train_s_label)

			# incremental train 
			print(f"Incremental training -->")
			#time.sleep(5)
			process.pre_trained = True
			process.g_episodic = e_net
			process.g_semantic = s_net
			e_net, s_net = process.run(re_train_data, re_train_e_label, re_train_s_label)

		elif args.continual_learning_mode == "NC":
			# train on frist 10 objects 
			# followed by incremental training on 5 objects per bacth
			intial_class_no = 10
			diff = 5
			flag = False
			process.batch = False
			for c in range(3):
				if flag == False:
					print("Training -->")
					time.sleep(2)
					e_net, s_net = process.run(data[:intial_class_no], e_label[:intial_class_no], s_label[:intial_class_no])
					flag = True

				# incremental train  - 5 objects per batch
				print("Incremental training -->")
				#time.sleep(2)
				process.pre_trained = True
				process.g_episodic = e_net
				process.g_semantic = s_net
				e_net, s_net = process.run(data[intial_class_no:intial_class_no+diff], e_label[intial_class_no:intial_class_no+diff], 
					s_label[intial_class_no:intial_class_no+diff])

				intial_class_no += diff

		elif args.continual_learning_mode == "NIC":
			s_size = int(500/args.bs_size)
			process.batch = False
			tmp_size = 0 

			# splitting instances (NI)
			data_to_scene = np.empty((int(data[0].shape[0]/s_size), data.shape[0]), dtype=object)
			e_label_to_scene = np.empty((int(e_label[0].shape[0]/s_size), e_label.shape[0]), dtype=object)
			s_label_to_scene = np.empty((int(s_label[0].shape[0]/s_size), s_label.shape[0]), dtype=object)
			
			for c in range(25):
				for d in range(len(data_to_scene)):
					data_to_scene[d][c] = data[c][tmp_size:tmp_size+s_size]
					e_label_to_scene[d][c] = e_label[c][tmp_size:tmp_size+s_size]
					s_label_to_scene[d][c] = s_label[c][tmp_size:tmp_size+s_size]
					tmp_size += s_size
				tmp_size = 0

			flag = False
			intial_class_no = 10
			diff = 5

			# NI with NC
			for d in range(len(data_to_scene)):
				if d == 0:
					for c in range(3):
						if flag == False:
							print(f"Training -->")
							print(e_label_to_scene[d][:intial_class_no])
							e_net, s_net = process.run(data_to_scene[d][:intial_class_no], e_label_to_scene[d][:intial_class_no], 
								s_label_to_scene[d][:intial_class_no])
							flag = True
						
						print("Incremental training -->")
						process.pre_trained = True
						process.g_episodic = e_net
						process.g_semantic = s_net
						''' uncomment below lines for debugging'''
						#print(data_to_scene[d][intial_class_no:intial_class_no+diff])
						#print(e_label_to_scene[d][intial_class_no:intial_class_no+diff])
						#print(s_label_to_scene[d][intial_class_no:intial_class_no+diff])
						e_net, s_net = process.run(data_to_scene[d][intial_class_no:intial_class_no+diff], 
							e_label_to_scene[d][intial_class_no:intial_class_no+diff], 
							s_label_to_scene[d][intial_class_no:intial_class_no+diff])

						intial_class_no += diff
				else:
					tmp_inital_class = 0 

					for _ in range(int(len(processing().obj_labels)/diff)):
						print("Incremental training -->")
						process.pre_trained = True
						process.g_episodic = e_net
						process.g_semantic = s_net
						
						e_net, s_net = process.run(data_to_scene[d][tmp_inital_class:tmp_inital_class+diff], 
							e_label_to_scene[d][tmp_inital_class:tmp_inital_class+diff], 
							s_label_to_scene[d][tmp_inital_class:tmp_inital_class+diff])

						tmp_inital_class += diff
		else:
			# run training
			e_net, s_net = process.run(data, e_label, s_label)

		if not args.pre_trained:
			# if using pre-trained network
			processing().export_network(save_folder+'/'+e_file, e_net)
			processing().export_network(save_folder+'/'+s_file, s_net)

			print(f"Episodic Network saved @ path -> {save_folder+'/'+e_file}")
			print(f"Semantic Network saved @ path -> {save_folder+'/'+s_file}")

	# testing
	if args.mode.lower() == 'both' or args.mode.lower() == 'test':
		e_net = processing().import_network(save_folder+'/'+e_file, EpisodicGWR)
		s_net = processing().import_network(save_folder+'/'+s_file, EpisodicGWR)
		
		print(f"Episodic Network loaded from path -> {save_folder+'/'+e_file}")
		print(f"Semantic Network loaded from path -> {save_folder+'/'+s_file}")
		
		if not args.no_tc_test:
			print(f"Testing without TC ----->")
			e_net.context = False
			s_net.context = False
			e_net.num_context = 0
			s_net.num_context = 0

		process.test(e_net=e_net,s_net=s_net,test_type='category', dist_type=args.dist_type, data_pre_process=args.preprocess, mode=args.test_mode.lower())
	
	else:
		raise Exception("Enter the correct mode of learning")


if __name__ == '__main__':

	working_dir = os.path.abspath(os.path.join(''))
	if working_dir not in sys.path:
		sys.path.append(working_dir)

	arg = argparse.ArgumentParser()
	# Learning
	arg.add_argument('--mode', type=str, default=None, help='Train/Test/Both')
	arg.add_argument('--test_mode', type=str, default='both', help='categorical/global/both')
	arg.add_argument('--learning_type', type=str, default=None, help="Batch/Incremental Learning")
	arg.add_argument('--continual_learning_mode', type=str, default=None, help="NI (new instance), NC (new class) & NIC (new instance and class)")
	arg.add_argument('--wandb_mode', type=str, default='online', help="online/offline")
	arg.add_argument('--features_path', type=str, default='features_256_negative_n_model_7', help="location of the features folder")
	arg.add_argument('--no_tc_test', type=bool, default=0, help="Enable/Disable Temporal Context during testing")

	# Pre-trained network
	arg.add_argument('--pre-trained', type=int, default=0, help="is network pre-trained (testing)")
	arg.add_argument('--episodic_network', type=str, default=None, help="Episodic Network location")
	arg.add_argument('--semantic_network', type=str, default=None, help="Semantic Network location")
	
	# Training
	arg.add_argument('--num_trials', type=int, default=1, help='total number of trials')
	arg.add_argument('--epochs', type=int, default=None, help="Number of train epochs (default, 25 - Batch, 3 - Incremental)")
	arg.add_argument('--dist_type', type=str, default="manhattan", help="distance type \n\
		- euclidean/l2_norm \
		- manhattan (best) \
		- minkowski \
		- manhalanobis \
		- cosine \
		")
	arg.add_argument('--preprocess', type=int, default=0, help="filter zeros")
	arg.add_argument('--mini_batch', type=int, default=1, help="split the dataset into mini batches")
	arg.add_argument('--bs_size', type=int, default=50, help="size of mini batches")

	# Hyperparameters
	arg.add_argument('--context', type=int, default=1, help="Enable Temporal Context")	
	arg.add_argument('--num_context', type=int, default=2, help="Number of context descriptors")
	arg.add_argument('--memory_replay', type=int, default=1, help="Enable Memory replay (incremental learning)")	
	arg.add_argument('--e_threshold', type=float, default=0.7, help="Episodic Memory insertion threshold")
	arg.add_argument('--s_threshold', type=float, default=0.8, help="Semantic Memory insertion threshold")
	arg.add_argument('--e_b', type=float, default=0.3, help="Learning rate - Epsilon BMU")
	arg.add_argument('--e_n', type=float, default=0.003, help="Learning rate -  Epsilon Neurons")
	arg.add_argument('--beta', type=float, default=0.5, help="Global Context")
	arg.add_argument('--e_regulated', type=int, default=0, help="Regulation type - Episodic Memory")
	arg.add_argument('--s_regulated', type=int, default=1, help="Regulation type - Semantic Memory")
	arg.add_argument('--habn_threshold', type=float, default=0.1, help="Habituation threshold")
	arg.add_argument('--node_rm_threshold', type=float, default=0.2, help="Neurons removal threshold")
	arg.add_argument('--max_age', type=int, default=2000, help="Maximum age of neurons in the network")

	# Output
	arg.add_argument('--output_folder', type=str, default=None, help="output folder name")

	args = arg.parse_args()
	parameters = {}
	count = 0
	
	for t in range(int(args.num_trials)):
		
		parameters['epochs'] = int(args.epochs)
		parameters['context'] = bool(args.context)
		parameters['num_context'] = int(args.num_context)
		parameters['memory_replay'] = bool(args.memory_replay)
		parameters['g_em_thresh'] = float(args.e_threshold)
		parameters['g_sm_thresh'] = float(args.s_threshold)
		parameters['beta'] = float(args.beta)
		parameters['e_b'] = float(args.e_b)
		parameters['e_n'] = float(args.e_n)
		parameters['e_regulated'] = int(args.e_regulated)
		parameters['s_regulated'] = int(args.s_regulated)
		parameters['habn_threshold'] = float(args.habn_threshold)
		parameters['node_rm_threshold'] = float(args.node_rm_threshold)
		parameters['max_age'] = int(args.max_age)

		run(args, parameters, t)

		
