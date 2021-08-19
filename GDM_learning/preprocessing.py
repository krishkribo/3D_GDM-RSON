
import os, sys
from time import time
from datetime import datetime
from copy import deepcopy

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
#
import numpy as np 
import cv2
from tqdm import tqdm
from PIL import Image
import pickle
#
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import open3d
# torch
import torch
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib._color_data as mcd
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import seuclidean
from scipy.spatial.distance import cdist

# To fix the "No Algorithm worked!" error uncomment below lines
"""config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
"""

plt.rcParams.update({'font.size': 18})

class model_preperation(object):

	def __init__(self):
		self.model = None

	def load_pre_trained(self, network_name=None, model_path=None, model_name=None, **kwargs):
		vis = kwargs.get('visual',False)
		m_type = kwargs.get('model_type', 'full')
		if network_name == 'grcnn':
			try:
				with open(model_path+'/'+model_name,'br') as m:
					model = torch.load(m)
					m.close()
			except Exception:
				tmp_path = ""
				model_path = model_path.split('/')[1:]
				for p in model_path:
					tmp_path += p+'/'

				model_path = tmp_path
				with open(model_path+'/'+model_name,'br') as m:
					model = torch.load(m)
					m.close()

			if m_type == 'full':
				model = model
			elif m_type == 'encoder':
				model = torch.nn.Sequential(*list(model.children())[:11])
			elif m_type == 'c_encoder' or m_type=='c_encoder_feature':
				model = torch.nn.Sequential(*list(model.children())[:12])    
			elif m_type == 'decoder':
				model = torch.nn.Sequential(*list(model.children())[13:])
			elif m_type == 'c_decoder':
				model = torch.nn.Sequential(*list(model.children())[12:])    
			
			if vis==True: print(model)

		return model


class processing(model_preperation):
	
	def __init__(self):
		self.dataset_path = "/home/krish/project_workspace/workspace/src/spawn_models/dataset/"
		self.data_folder = "data"
		self.model_path = '../CNN/generated_model/cornell'
		self.working_dir = os.path.abspath(os.path.join(''))
		self.no_of_categories = int(10/2)

		if self.working_dir not in sys.path:
			sys.path.append(self.working_dir)

		if "../CNN" not in sys.path:
			sys.path.append("../CNN")    

		#print(os.listdir())
		self.data = []
		self.objects = []
		self.obj_labels = ['airplane', 'staple', 'bowl', 'cooking_pan', 'pistol', 'guitar', 'pencil', 'wine_bottle', 'clock', 'hammer', 'car', 
							'spoon', 'jug', 'fork', 'earphone', 'chair', 'pen', 'knife', 'vase', 'water_bottle', 'mug', 'eraser', 'soda_can', 'usb', 'cereal_box']
		
		self.transform = transforms.Compose([
							transforms.ToTensor(),])
		self.device = torch.device('cuda')

	def write_file(self,file,value,path):
		# save the numpy array 
		if not os.path.exists(path):
			os.makedirs(path)
		with open(path+'/'+file,'wb') as f:
			np.save(f,np.array(value))
			print(f"Features stored at : {file} ")

	def load_file(self,file):
		# load features 
		with open(file,'rb') as f:
			value = np.load(f,allow_pickle=True)
			print("Data readed")
		return value    

	def import_network(self,file_name, NetworkClass):
		file = open(self.working_dir+'/'+file_name, 'br')
		data_pickle = file.read()
		file.close()
		net = NetworkClass()
		net.__dict__ = pickle.loads(data_pickle)
		return net
	
	def export_network(self,file_name, net):
		file = open(self.working_dir+'/'+file_name, 'wb')
		file.write(pickle.dumps(net.__dict__))
		file.close()

	def get_entropy(self, p_val):
		return -(p_val*np.log2(p_val))

	def get_entropy_img(self,img, gray_scale=False, norm=False):
		img = np.array(img)
		if gray_scale:
		   # convert the channel of 4 dim to 1 dim
		   g_img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
		   img = g_img
		if norm:
			n_val = np.linalg.norm(img)
			img = img/n_val
		h = 0 # entropy val
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				h += self.get_entropy(img[i,j])
		return h
	
	def get_distance(self, x, y, type='manhattan', **kwargs):
		p_val = kwargs.get('p_value', 3) # for minkowski distance
		x = np.array(x)
		y = np.array(y)
		if type == 'l2norm':
			return np.linalg.norm(x-y)
		elif type == 'euclidean':
			return np.sqrt(np.sum(np.square(x - y)))
		elif type == 'seuclidean':
			return np.sum(np.square(x - y))
		elif type == 'manhattan':
			return np.sum(np.linalg.norm(x-y,1))
		elif type == 'minkowski':
			return minkowski(x,y,p=p_val,w=None)
		elif type == 'mahalanobis':
			x = x.reshape(1,x.shape[0])
			y = y.reshape(1,y.shape[0])
			try:
				res = cdist(x.T, y.T, 'mahalanobis')
				d = np.diag(res)
				d = np.sum(d)/len(d) 
				return d
			except Exception:
				pass
		elif type=='cosine':
			d = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
			d = 1 - d # to get similarties based on least measure
			return d
		else:
			return None
	
	def filter_zeros(self, data):
		return np.delete(data, np.where(data == 0))

	def get_equal_array(self, **kwargs):
		x1 = np.array(kwargs.get('x1'))
		x2 = np.array(kwargs.get('x2'))

		l1 = len(x1)
		l2 = len(x2)
		if not l1 == l2:
			if l1 > l2:
				while l1 != l2 and l2<l1:
					x2 = np.append(x2, 0.0)
					l2 = len(x2)
			elif l2 > l1:
				while l2 != l1 and l1<l2:
					x1 = np.append(x1, 0.0)
					l1 = len(x1)
		return x1.reshape(-1,1), x2.reshape(-1,1)

	def show_image(self, data=None, **kwargs):
		pause_close = kwargs.get('pause_close',False)
		plt.figure(figsize=(10, 10))
		plt.imshow(data)
		if pause_close:
			plt.show(block=False)
			plt.pause(5)
			plt.close()
		else:
			plt.show()

	def get_rgb_depth_image(self, file_name, width=224, height=224):
		data = open3d.io.read_point_cloud(file_name)
		vis = open3d.visualization.Visualizer()
		vis.create_window('pcl', 720, 640, 50, 50, False)
		vis.add_geometry(data)
		# get the rgb and depth image from the point cloud
		depth_image = vis.capture_depth_float_buffer(True)
		rgb_image = vis.capture_screen_float_buffer(True)
		# convert to numpy array
		depth_image = np.asarray(depth_image)
		rgb_image = np.asarray(rgb_image)
		# save rgb and depth image to convert into pillow
		plt.imsave('output_depth.png',depth_image)
		plt.imsave('output_rgb.png',rgb_image)

		d_img = Image.open('output_depth.png')
		r_img = Image.open('output_rgb.png')

		return d_img.resize((width, height)), r_img.resize((width, height))

	def model_predict(self, data, **kwargs):
		m_type = kwargs.get('model_type', 'full')
		img_type = kwargs.get('img_type','quality')
		m_name = kwargs.get('model_name', None)
		
		model = super().load_pre_trained(network_name='grcnn', model_path=self.model_path, 
			model_name=m_name, model_type=m_type, visual=False)

		# preprocess images 
		if m_type != 'decoder' and m_type != 'c_decoder':
			try:
				img_transfrom = self.transform(data)
				img = torch.unsqueeze(img_transfrom,0)
			except Exception:
				img = data
		else:
			img = data

		model.eval()

		with torch.no_grad():

			if m_type == 'full':
				out = model[:11](img.to(self.device))
				out = model[13:](out)
				if img_type == 'quality':
					res = out[0]
					res = res.cpu().numpy().squeeze(1)
				elif img_type == 'angle':
					res = (torch.atan2(out[2], out[1]) / 2.0).cpu().numpy().squeeze(1)
				elif img_type == 'width':
					res = out[3]
					res = res.cpu().numpy().squeeze(1)*150
				res = res.reshape((res.shape[1],res.shape[2],res.shape[0]))

			elif m_type == 'encoder':
				out = model(img.to(self.device))
				res = out.cpu().numpy()[0]

			elif m_type == 'c_encoder':
				out = model(img.to(self.device))[0]
				res = out.cpu().numpy()[0]

			elif m_type == 'c_encoder_feature':
				out = model(img.to(self.device))[1]
				res = out.cpu().numpy()[0]

			elif m_type == 'decoder' or m_type == 'c_decoder':
				try:
					out = model(img)
				except Exception:
					out = model(img.to(self.device))

				if img_type == 'quality':
					res = out[0]
					res = res.cpu().numpy().squeeze(1)
				elif img_type == 'angle':
					res = (torch.atan2(out[2], out[1]) / 2.0).cpu().numpy().squeeze(1)
				elif img_type == 'width':
					res = out[3]
					res = res.cpu().numpy().squeeze(1)*150

				try:
					res = res.reshape((res.shape[1],res.shape[2]))
				except Exception:
					res = res.reshape((res.shape[1],res.shape[2],res.shape[0]))

		return out, res

	def get_minima(self, img, offset, **kwargs):
		''' 
		mulitiple grasp pts - 
		based on local minimum peaks and
		distance between the neighbours
		'''
		show = kwargs.get('show', False)
		save = kwargs.get('save', False)
		n_grasp = kwargs.get('n_grasp', 1)
		min_dist = kwargs.get('min_dist', 10)
		threshold = kwargs.get('thresh', 0.25)

		img = np.array(img)
		width = img.shape[0]
		height = img.shape[1]

		# filter borders
		img = img[offset:width-offset, :]
		img = img[:, offset:height-offset]

		# copy the image
		img1 = img  

		# get the min and max value of the input
		g_min_val, g_max_val, _, _ = cv2.minMaxLoc(img)
		min_indx_list = []
		min_val_list = []
		count = 0
		g_flag = False

		for g in range(img.shape[0]*img.shape[1]):
			# get the local minima and maxima
			min_val,max_val,min_indx,max_indx=cv2.minMaxLoc(img1)
			if min_val >= threshold:
				if [min_indx[1], min_indx[0]] not in min_indx_list:
					#print(f"min diff val : {int(min_val*100 - g_min_val*100)}")
					if (min_val - g_min_val)*100 >= min_dist or g == 0:
						min_indx_list.append([min_indx[1], min_indx[0]])
						min_val_list.append(min_val)
						g_min_val = min_val
						count+=1
						if count == n_grasp:
							# all grasps are obtained
							g_flag = True

			img1[min_indx[1], min_indx[0]] += g_max_val

			if g_flag:
				# all grasps are obtained
				break

		# draw the grasp point
		for g in min_indx_list:
			cv2.circle(img, (g[1], g[0]), 1, (0,0,0), 2)	
			#pass
		
		if show:
			cv2.imshow('im', img)
			if cv2.waitKey(0) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
		if save:
			cv2.imwrite("grasp_res/raw_{}.png".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), img*255)

		return min_indx_list

	def get_data(self, pcl_file=None, width=224, height=224):
		depth_image, rgb_image = self.get_rgb_depth_image(pcl_file, width=width, height=height)
		depth_image = np.array(depth_image)
		rgb_image = np.array(rgb_image)

		return depth_image, rgb_image

	def get_data_batch(self, width=224, height=224, get_angles=True):
		
		if self.data_folder not in os.listdir():
			os.mkdir(self.working_dir+'/'+self.data_folder)

		for s in tqdm(os.listdir(self.dataset_path)):
			for d in tqdm(os.listdir(self.dataset_path+'/'+s)):
				self.objects.append(d)
				for pcl_file in os.listdir(self.dataset_path+'/'+s+'/'+d):
					depth_image, rgb_image = self.get_rgb_depth_image(self.dataset_path+'/'+s+'/'+d+'/'+pcl_file, width=width, height=height)
					depth_image = np.array(depth_image)
					rgb_image = np.array(rgb_image)
					e_labels = self.objects.index(d) # instance level labels
					s_labels = self.obj_labels.index(self.obj_labels[int(e_labels)]) # category level labels
					if get_angles:
						roll = pcl_file.split('_')[3]
						pitch = pcl_file.split('_')[4]
						yaw = pcl_file.split('_')[5]
						self.data.append([depth_image, rgb_image, roll, pitch, yaw, e_labels, s_labels])
					else:	
						self.data.append([depth_image, rgb_image, e_labels, s_labels])
			self.objects = []
			
			time.sleep(0.5) # sleep for 5 secs
			self.write_file(s+'_data.npy',self.data, self.working_dir+'/'+self.data_folder+'/')
			self.data = []

	def get_feature(self, data=None, norm = False, model_name=None):
		features_predicted = self.model_predict(data, model_type='c_encoder_feature', model_name=model_name)[1]
		if norm:
			print("features normalized --->")
			features_predicted_edt = normalize(features_predicted.reshape(1, -1))   
		else: 
			features_predicted_edt = features_predicted 
		print(f"Shape of the features predicted : {features_predicted_edt.shape}")
		return features_predicted_edt

	def get_features_batch(self, norm = False, model_name=None):
		features_folder = "tmp_features"

		if features_folder not in os.listdir():
			os.mkdir(self.working_dir+'/'+features_folder)

		for s in tqdm(os.listdir(self.data_folder)):
			scene_no = s.split('_')[0]
			# load data and labels 
			data = self.load_file(self.data_folder+'/'+s)
			features_predicted = np.array([np.array([self.model_predict(d[0], model_type='c_encoder_feature', model_name=model_name)[1]
				,d[-2],d[-1]],dtype=object) for d in tqdm(data)])

			print(f"Features shape :{features_predicted.shape}")
			print(features_predicted[0][0].shape)
			if norm:
				print("features normalized --->")
				features_predicted_edt = np.array([np.array([normalize(f[0].reshape(1, -1)),f[1],f[2]],dtype=object) for f in tqdm(features_predicted)])     
			else:
				print("features not normalized --->")
				features_predicted_edt = np.array([np.array([f[0].reshape(1, -1),f[1],f[2]],dtype=object) for f in tqdm(features_predicted)])
			
			#features_predicted_edt = np.array(features_predicted_edt)
 
			print(f"Shape of the features predicted : {features_predicted_edt.shape}")
			self.write_file(scene_no+'_feature.npy',features_predicted_edt,self.working_dir+'/'+features_folder)

	def transform_features_batch(self, f_name=None, output_size=256, norm=True):
		features_folder = f_name

		if features_folder not in os.listdir():
			os.mkdir(self.working_dir+'/'+features_folder)

		# load CNN model
		pca = PCA(n_components=output_size)

		for s in tqdm(os.listdir(f_name)):
			scene_no = s.split('_')[0]
			if scene_no+'_feature.npy' not in os.listdir(features_folder):
				# load data and labels 
				features_predicted = self.load_file(f_name+'/'+s)
				features = np.array([f[0] for f in tqdm(features_predicted)])
				pca.fit(features) 
				features = []
				if not output_size is None:
					print(f"features transformed to : {output_size}")
					if norm:
						features_predicted_edt = np.array([np.array([normalize(pca.transform(f[0].reshape(1,-1))),f[1],f[2]],dtype=object) for f in tqdm(features_predicted)])
					else:    
						features_predicted_edt = np.array([np.array([pca.transform(f[0].reshape(1,-1)),f[1],f[2]],dtype=object) for f in tqdm(features_predicted)])
				else:
					features_predicted_edt = features_predicted
	 
				print(f"Shape of the features predicted : {features_predicted_edt.shape}")
				self.write_file(scene_no+'_feature.npy',features_predicted_edt, self.working_dir+'/'+features_folder+'/')
				features_predicted_edt = []
				features_predicted = []

	def plot_network(self, n_net, edges, labels,title,network=None, **kwargs):
		show_plot = kwargs.get('show', False)
		context = kwargs.get('contex', None)
		print(f"----------->")
		net = deepcopy(n_net)
		color = list({name for name in mcd.CSS4_COLORS if "xkcd:" + name in mcd.XKCD_COLORS})
		plt.figure(figsize=(15, 15))
		print(f"length of net weights : {len(net.weights[0].shape)}")
		if len(net.weights[0].shape) < 2:
			dim_net = True 
		else:
			dim_net = False
			# added PCA
			if context:
				pca = PCA(n_components=2)
				for n in range(len(net.weights)):
					pca.fit(net.weights[n])
					net.weights[n] = pca.transform(net.weights[n])
		###
		num_connections = 0
		for ni in range(len(net.weights)):

			if network == "episodic":
				plindex =  int(np.argmax(net.alabels[0][ni]))
			elif network == "semantic":
				plindex = np.argmax(net.alabels[0][ni])

			if labels:
				if dim_net:
					plt.scatter(net.weights[ni][0], net.weights[ni][1], color=color[plindex], alpha=.5)
					plt.text(net.weights[ni][0], net.weights[ni][1], 'C'+str(plindex))
				else:
					plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], color=color[plindex], alpha=.5)
					plt.text(net.weights[ni][0,0], net.weights[ni][0,1], 'C'+str(plindex))
			else:
				if dim_net:
					plt.scatter(net.weights[ni][0], net.weights[ni][1], alpha=.5)
					plt.text(net.weights[ni][0], net.weights[ni][1], 'C'+str(plindex))
				else:
					plt.scatter(net.weights[ni][0, 0], net.weights[ni][0, 1], alpha=.5)
					plt.text(net.weights[ni][0], net.weights[ni][1], 'C'+str(plindex))
			if edges:
				for nj in range(len(net.weights)):
					if net.edges[ni, nj] > 0:
						if dim_net:
							plt.plot([net.weights[ni][0], net.weights[nj][0]], 
									 [net.weights[ni][1], net.weights[nj][1]],
									 'gray', alpha=.3)
						else:
							plt.plot([net.weights[ni][0, 0], net.weights[nj][0, 0]], 
									 [net.weights[ni][0, 1], net.weights[nj][0, 1]],
									 'gray', alpha=.3)                        
						num_connections+=1
					
		print(f"num of connections : {num_connections}")
		plt.title(title)
		plt.xlabel('Network weights')
		plt.ylabel('Network weights')
		if show_plot:
			plt.show(block=False)
			plt.pause(5)
		
		print("---------->")
		return plt

	def plot_data(self,value,title,l_type,y_label, path, **kwargs):

		save = kwargs.get('save_fig', True)
		plt_type = kwargs.get('plt_type', None)
		episode = str(kwargs.get('episode', 0))
		legend = kwargs.get('legend', None)
		instance_acc_data = kwargs.get('i_data', None)
		category_acc_data = kwargs.get('c_data', None)
		show = kwargs.get('show', False)
		fname = kwargs.get('file_name', False)

		color = ['r', 'g', 'b'] # for 3 scene, expand the colors based on new test scenes

		plt.figure(figsize=(14, 14))
		if plt_type == 'single':
			plt.plot(value)

		elif plt_type == 'multiple':
			for i in value:
				plt.plot(i)
			plt.legend(legend,loc='best')
			

		elif plt_type == 'multiple_bar_both':
			x_range = np.arange(len(value))
			plt.bar(x_range[:-1] - 0.2, instance_acc_data[:-1], 0.2, label = "Instance level", color='blue')
			plt.bar(x_range[:-1] + 0.2, category_acc_data[:-1], 0.2, label = "Category level", color='red')
			plt.bar(x_range[-1] - 0.2, instance_acc_data[-1], 0.2, label = "Instance level - Average", color='grey')
			plt.bar(x_range[-1] + 0.2, category_acc_data[-1], 0.2, label = "Category level - Average", color='black')

			plt.xticks(x_range, value, rotation='vertical')
			plt.legend()

		elif plt_type == 'multiple_bar_category':
			x_range = np.arange(len(value))
			val = -0.2

			for j in range(len(category_acc_data)):
				plt.bar(x_range[:-1] + val, category_acc_data[j][:-1], 0.2, label = "Category level - "+str(legend[j]), color=color[j])
				val += 0.2
			val = -0.2

			for j in range(len(category_acc_data)):
				plt.bar(x_range[-1] + val, category_acc_data[j][-1], 0.2, label = "(AVG) Category level - "+str(legend[j]), color=color[j])
				val += 0.2
			
			plt.xticks(x_range, value, rotation='vertical')
			plt.legend()
		
		if l_type  == 0:
			x_label = "Epochs"
		elif l_type == 1:
			x_label = "Number of Learned Categories"	
		else:
			x_label=""

		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.title(title)
		plt.grid(True, axis='both', color='gray', linestyle='--', linewidth=0.2)
		if save:
			if not os.path.exists(path):
				os.makedirs(path)
			time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
			plt.savefig(path+'/'+fname+'_episode_'+str(episode)+"_{}.png".format(time))
		if show:
			plt.show()

	def plot_grasp(self, fig, rgb_img, depth_img, q_img, angle_img, width_img, grasp_pts, offset_val, **kwargs):
		save = kwargs.get('save', False)
		show = kwargs.get('show', False)

		plt_type = kwargs.get('plt_type', 'full')

		plt.clf()
		plt.ion()	

		if plt_type == 'full':
			ax = fig.add_subplot(2, 3, 1)
			ax.imshow(rgb_img)
			ax.set_title('RGB')
			ax.axis('off')

			if depth_img is not None:
				ax = fig.add_subplot(2, 3, 2)
				ax.imshow(depth_img, cmap='gray')
				ax.set_title('Depth')
				ax.axis('off')

			ax = fig.add_subplot(2, 3, 3)
			ax.imshow(rgb_img)

			for g in grasp_pts:
				g.plot(ax, offset_val)

			ax.set_title('Grasp')
			ax.axis('off')
			
			ax = fig.add_subplot(2, 3, 1)
			plot = ax.imshow(q_img, cmap='jet', vmin=0, vmax=1)
			ax.set_title('Q')
			ax.axis('off')
			plt.colorbar(plot)

			ax = fig.add_subplot(2, 3, 2)
			plot = ax.imshow(angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
			ax.set_title('Angle')
			ax.axis('off')
			plt.colorbar(plot)

			ax = fig.add_subplot(2, 3, 3)
			plot = ax.imshow(width_img, cmap='jet', vmin=0, vmax=100)
			ax.set_title('Width')
			ax.axis('off')
			plt.colorbar(plot)

			plt.pause(0.1)
			fig.canvas.draw()
			if show: plt.show()

		else:
			ax = plt.subplot(111)

			if plt_type == 'rgb':
				ax.imshow(rgb_img)
				ax.set_title('RGB')
				ax.axis('off')

			elif plt_type == 'depth':
				ax.imshow(depth_img, cmap='gray')
				ax.set_title('Depth')
				ax.axis('off')

			elif plt_type == 'grasp':
				ax.imshow(rgb_img)

				for g in grasp_pts:
					g.plot(ax, offset_val)

				ax.set_title('Grasp')
				ax.axis('off')

			elif plt_type == 'q_img':
				plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
				ax.set_title('Q')
				ax.axis('off')
				plt.colorbar(plot)

			elif plt_type == 'angle_img':
				plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
				ax.set_title('Angle')
				ax.axis('off')
				plt.colorbar(plot)

			elif plt_type == 'width_img':
				plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
				ax.set_title('Width')
				ax.axis('off')
				plt.colorbar(plot)

		if save:
			time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
			fig.savefig('grasp_res/results_f_{}.png'.format(time))


if __name__ == "__main__":
	import argparse
	processing = processing()

	arg = argparse.ArgumentParser(description="Enter the type of preprocessing")
	arg.add_argument('--get_data', metavar='-D', type=bool, default=False, help="Get the data from the dataset")
	arg.add_argument('--get_features', metavar='-F', type=bool, default=False, help="Get the features from the converted data")	
	arg.add_argument('--transform_features', metavar='-T', type=bool, default=False, help="Transform the features using PCA")
	arg.add_argument('--visual', metavar='-V', type=bool, default=False, help="print or plot the data")
	arg.add_argument('--model_name', metavar='-M', type=str, default=None, help="enter the model name used for feature extraction")
	args = arg.parse_args()
	
	if args.get_data:
		processing.get_data_batch(width=224, height=224, get_angles=True)

	if args.get_features:
		processing.get_features_batch(norm=True, model_name=args.model_name)

	if args.transform_features:
		processing.transform_features_batch(f_name='features', output_size=256, norm=True)

	if args.visual:
		data = processing.load_file("tmp_features/s1_feature.npy")
		print(data.shape)
		print(data[0][0].shape)
		print(data[0])
		print(data[0][0].shape)
		plt.imshow(Image.fromarray(np.array(data[500][0])))
		plt.show()