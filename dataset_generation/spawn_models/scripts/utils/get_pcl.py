#! /usr/bin/env python3

import open3d
from ctypes import *
from sensor_msgs.msg import PointCloud2,PointField
from sensor_msgs import point_cloud2
import numpy as np
import rospy
from copy import deepcopy


class pcl_helper(object):
    """
    get the data from the point cloud publisher and save to .pcd file 
    """
    def __init__(self):

        """
        source : https://github.com/felixchenfy/open3d_ros_pointcloud_conversion/blob/master/lib_cloud_conversion_between_Open3D_and_ROS.py
        """
        self.convert_rgbUint32_to_tuple = lambda rgb_uint32: (
            (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
        )
        self.convert_rgbFloat_to_tuple = lambda rgb_float: self.convert_rgbUint32_to_tuple(
            int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
        )
        self.binary = None
        self.filename = None
        self.pcl_data = None

    def add_gauss_noise(self, data=None, mu=None, sigma=None):
        if not (mu == None or mu == 0 and sigma == None or sigma == 0):
            pcl_copy = deepcopy(data)
            points = np.asarray(pcl_copy.points)
            points += np.random.normal(mu, sigma, size=points.shape)
            pcl_copy.points = open3d.utility.Vector3dVector(points)

            return pcl_copy

    def downsample(self, data=None, percent=50):
        if not percent == None or percent == 0:
            ds_size = ((percent/100)*len(data.points))/len(data.points)
            return data.voxel_down_sample(voxel_size=ds_size)

    def read_and_convert(self, data=None, add_noise=False, mu=None, sigma=None, downsample=False, ds_percent=50):
        # get the field names
        field_names=[field.name for field in data.fields]
        read_data = point_cloud2.read_points(data,skip_nans=True,field_names=field_names)
        cloud_data = list(read_data)
        #rospy.loginfo("point cloud data obtained")
        
        # condition to add rgb data
        if self.binary == True:
            field_names.remove(field_names[-1])
        else:
            field_names = field_names        

        if len(list(cloud_data))==0:
            print("empty cloud")
            return None
            exit()
        
        # open3d
        #rospy.loginfo("open 3d point cloud generation -->")
        open3d_cloud = open3d.geometry.PointCloud()
        
        if "rgb" in field_names:
            #rospy.loginfo("PCL fields : xyz rgb")
            IDX_RGB_IN_FIELD=3 # x, y, z, rgb
            
            # Get xyz
            xyz = [(x,y,z) for x,y,z,rgb in cloud_data ]

            # Get rgb
            # Check whether int or float
            if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
                rgb = [self.convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
            else:
                rgb = [self.convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]

            # combine
            open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))
            open3d_cloud.colors = open3d.utility.Vector3dVector(np.array(rgb)/255.0)
        else:
            #rospy.loginfo("PCL fields : xyz")
            xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
            open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))
        
        # write pcl
        if add_noise:
            # add gaussian noise
            noise_data = self.add_gauss_noise(data=open3d_cloud, mu=mu, sigma=sigma)   
            open3d.io.write_point_cloud(self.filename, noise_data)
        elif downsample:
            # downsample point cloud 
            ds_data = self.downsample(data=open3d_cloud, percent=ds_percent)
            #print(len(np.array(ds_data.points)))
            if len(np.array(ds_data.points)) < 10:
                raise Exception(f"point cloud  points less than 10 (no of points : {ds_data.points} \
                    ds_percent : {ds_percent})")
            open3d.io.write_point_cloud(self.filename, ds_data)
        elif add_noise and downsample:
            # add gaussian noise
            noise_data = self.add_gauss_noise(data=open3d_cloud, mu=mu, sigma=sigma) 
            # downsample point cloud
            ds_data = self.downsample(data=noise_data, percent=ds_percent)
            if len(np.array(ds_data.points)) < 10:
                raise Exception(f"point cloud  points less than 10 (no of points : {ds_data.points} \
                    ds_percent : {ds_percent})")
            open3d.io.write_point_cloud(self.filename, ds_data)
        else:    
            open3d.io.write_point_cloud(self.filename, open3d_cloud)

        #rospy.loginfo("Writed point cloud file :"+self.filename)
        
    def run_pcl(self, data, filename, binary=False, add_noise=False, mu=None, sigma=None, downsample=False, ds_percent=None):
        self.binary = binary
        self.filename = filename
        self.read_and_convert(data=data, add_noise=add_noise, mu=mu, sigma=sigma, downsample=downsample, ds_percent=ds_percent)         

if __name__ == "__main":
    rospy.init_node("node_pcl",anonymous=True)
    pcl = pcl_helper()
    res= rospy.Subscriber("/point_cloud_functions/cloud", PointCloud2, pcl.read_and_convert, queue_size=1)
    rospy.spin()
    
        


