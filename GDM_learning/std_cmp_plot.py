import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from preprocessing import processing

plt.rcParams.update({'font.size': 20})


working_dir = os.path.abspath(os.path.join(''))
if not working_dir in sys.path:
	sys.path.append(working_dir)


def load_data(file):
	return np.load(file)

def plot_data(data=None, path=None, fname=None, **pargs):
	show = pargs.get('show', False)
	save = pargs.get('save', False)
	legend = pargs.get('legend', None)
	l_type = pargs.get('l_type', None)
	y_label = pargs.get('y_label', None)
	title = pargs.get('title', None)
	
	plt.figure(figsize=(9, 9))
	
	data = np.array(data)
	
	if l_type == 1 and title == 'Accuracy(%)':
		x_val = np.arange(len(data[0][0]))+1
	else:
		x_val = np.arange(len(data[0][0]))

	mean_val = []
	std_val = []

	for d in range(len(data)):
		
		mean_data = np.mean(data[d], axis=0)
		std_data = np.std(data[d], axis=0)

		plt.plot(x_val,mean_data)

		plt.fill_between(x_val, mean_data-std_data, mean_data+std_data, alpha=0.3)

		mean_val.append(mean_data)
		std_val.append(std_data)

	if l_type == 0:
		x_label = "Epochs"
	elif l_type == 1:
		x_label = "Number of Learned Categories"
	else:
		x_label = ""

	plt.xlabel(x_label, fontsize=24)
	plt.ylabel(y_label, fontsize=24)
	plt.title(title)
	plt.legend(legend,loc='best')
	plt.grid(True, axis='both', color='gray', linestyle='--', linewidth=0.2)
	plt.tight_layout()

	if save:
		if not os.path.exists(path):
			os.makedirs(path)
		plt.savefig(path+'/'+fname+".png")
		print(f"Plot save @ {path+'/'+fname}.png")
	
	if show:	
		plt.show()
		plt.close()

	return mean_val, std_val

if __name__ == "__main__":
	arg = argparse.ArgumentParser()
	arg.add_argument('--path1', type=str, default=None, help="Enter the location of experiments to get std plot for without replay/TC")
	arg.add_argument('--path2', type=str, default=None, help="Enter the location of experiments to get std plot for with replay/TC")
	arg.add_argument('--learning_type', type=int, default=None, help="Learning type, Batch - 0, Incremental - 1")

	args = arg.parse_args()
	episodic_acc_list = [[],[]]
	semantic_acc_list = [[],[]]
	episodic_neurons_list = [[],[]]
	semantic_neurons_list = [[],[]]

	paths = [args.path1, args.path2]
	for p in range(len(paths)):
		for dir in os.listdir(working_dir+'/'+paths[p]):
			if os.path.isdir(working_dir+'/'+paths[p]+'/'+dir):
				for sub_dir in os.listdir(working_dir+'/'+paths[p]+'/'+dir):
					check_npy = sub_dir.split('.')

					if check_npy[-1] == 'npy':
						check_file = sub_dir.split('_')
						
						if check_file[1] == 'accuracy':
							acc_data = load_data(working_dir+'/'+paths[p]+'/'+dir+'/'+sub_dir)
							if args.learning_type == 1:
								if check_file[0] == 'episodic':
									episodic_acc_list[p].append(acc_data[1:])

								elif check_file[0] == 'semantic':
									semantic_acc_list[p].append(acc_data[1:])
							else:
								if check_file[0] == 'episodic':
									episodic_acc_list[p].append(acc_data)

								elif check_file[0] == 'semantic':
									semantic_acc_list[p].append(acc_data)


						if check_file[1] == 'neuron':
							n_data = load_data(working_dir+'/'+paths[p]+'/'+dir+'/'+sub_dir)

							if check_file[0] == 'episodic':
								episodic_neurons_list[p].append(n_data)

							elif check_file[0] == 'semantic':
								semantic_neurons_list[p].append(n_data)

	# plot accuracy	
	plot_data(data=episodic_acc_list, path=working_dir+'/'+'comparison_results', fname="std_em_accuracy_plot", l_type=args.learning_type, y_label="Accuracy(%)", 
		legend=["G-EM without replay", "G-EM with replay"], title="Accuracy(%)", show=False, save=True)
	plot_data(data=semantic_acc_list, path=working_dir+'/'+'comparison_results', fname="std_sm_accuracy_plot", l_type=args.learning_type, y_label="Accuracy(%)", 
		legend=["G-SM without replay", "G-SM with replay"], title="Accuracy(%)", show=False, save=True)					
	plot_data(data=episodic_neurons_list, path=working_dir+'/'+'comparison_results', fname="std_em_neuron_plot", l_type=args.learning_type, y_label="Neurons", 
		legend=["G-EM without replay", "G-EM with replay"], title="No. of Neurons", show=False, save=True)	
	plot_data(data=semantic_neurons_list, path=working_dir+'/'+'comparison_results', fname="std_sm_neuron_plot", l_type=args.learning_type, y_label="Neurons", 
		legend=["G-SM without replay", "G-SM with replay"], title="No. of Neurons", show=False, save=True)	
	
	



