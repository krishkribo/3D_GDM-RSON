import open3d as o3d 
import numpy as np
import os 
import sys 


working_dir = os.path.abspath(os.path.join('.'))
if working_dir not in sys.path:
	sys.path.append(working_dir)

scene_dir = 'dataset'

for s in os.listdir(working_dir+'/'+scene_dir):
	for obj in os.listdir(working_dir+'/'+scene_dir+'/'+s):
		for d in os.listdir(working_dir+'/'+scene_dir+'/'+s+'/'+obj):
			pcl_pts = o3d.io.read_point_cloud(working_dir+'/'+scene_dir+'/'+s+'/'+obj+'/'+d)
			l_pcl_pts = len(np.asarray(pcl_pts.points))
			if l_pcl_pts < 10:
				raise Exception(f"pcl points less than 10 : {l_pcl_pts}, scene: {s}, object : {obj}, file: {d}")
			else:
				print(f"scene: {s} :: object: {obj} :: file: {d} --> points: {l_pcl_pts}")
