#! /usr/bin/env python3

import open3d
from ctypes import *
from sensor_msgs.msg import PointCloud2,PointField
from sensor_msgs import point_cloud2
import numpy as np
import rospy


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

    def read_and_convert(self,pcl_data):
        # get the field names
        field_names=[field.name for field in pcl_data.fields]
        read_data = point_cloud2.read_points(pcl_data,skip_nans=True,field_names=field_names)
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
        open3d.io.write_point_cloud(self.filename,open3d_cloud)
        #rospy.loginfo("Writed point cloud file :"+self.filename)
        
    def run_pcl(self, data, filename, binary=False):
        self.binary = binary
        self.filename = filename
        self.read_and_convert(data)         

if __name__ == "__main":
    rospy.init_node("node_pcl",anonymous=True)
    pcl = pcl_helper()
    res= rospy.Subscriber("/point_cloud_functions/cloud", PointCloud2, pcl.read_and_convert, queue_size=1)
    rospy.spin()
    
        


