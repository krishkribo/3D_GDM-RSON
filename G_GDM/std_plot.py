import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse

plt.rcParams.update({'font.size': 14})


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
	
	plt.figure(figsize=(10, 10))

	x_val = np.arange(len(data[0][0]))
	mean_val = []
	for d in data:
		mean_data = np.mean(d, axis=0)
		std_data = np.std(d, axis=0)

		#print(f"mean: ",mean_data)
		#print(f"std: ", std_data)

		plt.plot(x_val,mean_data)
		plt.fill_between(x_val, mean_data-std_data, mean_data+std_data, alpha=0.3)

		mean_val.append(mean_data)

	if l_type == 0:
		x_label = "Epochs"
	elif l_type == 1:
		x_label = "Number of Learned Categories"
	else:
		x_label = ""

	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.legend(legend,loc='best')
	plt.grid(True, axis='both', color='gray', linestyle='--', linewidth=0.2)

	if save:
		if not os.path.exists(path):
			os.makedirs(path)
		plt.savefig(path+'/'+fname+".png")
		print(f"Plot save @ {path+'/'+fname}.png")
	
	if show:	
		plt.show()
		plt.close()

	return mean_val

	

if __name__ == "__main__":
	arg = argparse.ArgumentParser()
	arg.add_argument('--path', type=str, default=None, help="Enter the location of experiments to get std plot")
	arg.add_argument('--learning_type', type=int, default=None, help="Learning type, Batch - 0, Incremental - 1")

	args = arg.parse_args()
	#print(os.listdir(args.path))
	acc_list = [[],[]]
	err_list = [[],[]]
	neurons_list = [[],[]]
	update_rate_list = [[],[]]

	for dir in os.listdir(working_dir+'/'+args.path):
		if os.path.isdir(working_dir+'/'+args.path+'/'+dir):
			for sub_dir in os.listdir(working_dir+'/'+args.path+'/'+dir):
				check_npy = sub_dir.split('.')

				if check_npy[-1] == 'npy':
					check_file = sub_dir.split('_')
					
					if check_file[1] == 'accuracy':
						acc_data = load_data(working_dir+'/'+args.path+'/'+dir+'/'+sub_dir)
						
						if check_file[0] == 'episodic':
							acc_list[0].append(acc_data)

						elif check_file[0] == 'semantic':
							acc_list[1].append(acc_data)

					if check_file[1] == 'error':
						err_data = load_data(working_dir+'/'+args.path+'/'+dir+'/'+sub_dir)
						
						if check_file[0] == 'episodic':
							err_list[0].append(err_data)

						elif check_file[0] == 'semantic':
							err_list[1].append(err_data)

					if check_file[1] == 'neuron':
						n_data = load_data(working_dir+'/'+args.path+'/'+dir+'/'+sub_dir)
						
						if check_file[0] == 'episodic':
							neurons_list[0].append(n_data)

						elif check_file[0] == 'semantic':
							neurons_list[1].append(n_data)

					if check_file[1] == 'update':
						r_data = load_data(working_dir+'/'+args.path+'/'+dir+'/'+sub_dir)
						
						if check_file[0] == 'episodic':
							update_rate_list[0].append(r_data)

						elif check_file[0] == 'semantic':
							update_rate_list[1].append(r_data)


	# plot accuracy	
	mean_acc = plot_data(data=acc_list, path=working_dir+'/'+args.path, fname="std_accuracy_plot", l_type=args.learning_type, y_label="Accuracy", 
		legend=["Instance level", "Category level"], title="Accuracy(%)", show=False, save=True)					
	# plot quantization error
	plot_data(data=err_list, path=working_dir+'/'+args.path, fname="std_qerr_plot", l_type=args.learning_type, y_label="ATQE", 
		legend=["G-EM", "G-SM"], title="Quantization Error", show=False, save=True)	
	# plot number of neurons
	plot_data(data=neurons_list, path=working_dir+'/'+args.path, fname="std_neuron_plot", l_type=args.learning_type, y_label="Neurons", 
		legend=["G-EM", "G-SM"], title="No. of Neurons", show=False, save=True)	
	# plot update rate
	plot_data(data=update_rate_list, path=working_dir+'/'+args.path, fname="std_update_rate_plot", l_type=args.learning_type, y_label="Rate", 
		legend=["G-EM", "G-SM"], title="Update Rate", show=False, save=True)	


	
	# get the best performing trials - based on accuracy on categgory level (0, max_trial)
	category_acc = acc_list[1]
	best_acc = [np.sum(d) for d in category_acc]
	print(f"Best accuracy @ trial {np.argmax(best_acc)}")
	print(f"Accuracy values : {category_acc[np.argmax(best_acc)]}")

	print(f"Average accuracy accross learning trials : {mean_acc[0][-1], mean_acc[1][-1]}")




