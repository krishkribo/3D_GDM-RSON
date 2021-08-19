#! /usr/bin/env python3

import rospy
from point_cloud_functions.srv import GetObjectROI, GetObjectROIRequest
import numpy as np


class get_object_roi(object):
    """
    Get the object region of interest
    """
    def __init__(self):
        self.roi_client = rospy.ServiceProxy("get_object_roi", GetObjectROI)
        rospy.loginfo("Waiting for /get_object_roi service")
        self.roi_client.wait_for_service()
        rospy.loginfo("Connected to /get_object_roi service")
        self.flag = False

    def run(self):
        self.roi_msg = GetObjectROIRequest()
        self.roi_msg.transform_to_link = "new_frame_id"
        self.roi_msg.surface_threshold = 0.010
        self.roi_msg.cluster_distance = 0.05
        self.roi_msg.filter_x = True
        self.roi_msg.min_x = -2.5
        self.roi_msg.max_x = 2.5
        self.roi_msg.filter_y = True
        self.roi_msg.min_y = -2.5
        self.roi_msg.max_y = 2.5
        self.roi_msg.filter_z = False
        self.roi_msg.min_cluster_points = 100
        self.roi_msg.max_cluster_points = 50000
        self.roi_msg.voxelize_cloud = True
        self.roi_msg.leaf_size = 0.005

        try:
            result_rois = self.roi_client.call(self.roi_msg)
        except:
            print("Waiting for image") 

    def get_roi(self):
        self.run()
        try:
            result = self.roi_client(self.roi_msg)
        except:
            return

        return result.object_rois
