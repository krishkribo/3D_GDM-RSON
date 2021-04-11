import numpy as np
import rospy
from point_cloud_functions.srv import GetObjectROI, GetObjectROIRequest
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
import cv2
import time
import sys
import os
label = sys.argv[1]

DATADIR = '/home/borg/dataset/' + label + '/'

if not os.path.exists(DATADIR):
    os.mkdir(DATADIR)
    
def point_cloud_callback(data):
  pass

def image_callback(data):
  pass
        
print "Annotating ", label

point_cloud_topic = "/xtion/depth_registered/points"
image_topic = "xtion/rgb/image_raw" 

rospy.init_node("object_annotation")
get_object_roi_client = rospy.ServiceProxy("/get_object_roi", GetObjectROI)
bridge = CvBridge()

point_cloud_subscriber = rospy.Subscriber(point_cloud_topic, PointCloud2, point_cloud_callback)
image_subscriber = rospy.Subscriber(image_topic, Image, image_callback)
rospy.sleep(1.0)

print "Waiting for /get_object_roi service"
get_object_roi_client.wait_for_service()
print "Connected to /get_object_roi service"


imageIdx = 0
while True:

    request = GetObjectROIRequest()
    request.transform_to_link = "base_link"
    request.cluster_distance  = 0.05
    request.surface_threshold = 0.015
    request.filter_x = True
    request.min_x = 0.1
    request.max_x = 0.8
    request.filter_y = True
    request.min_y = -0.3
    request.max_y = 0.3
    request.filter_z = True
    request.min_z = 0.3
    request.max_z = 2.0
    request.min_cluster_points = 10
    request.max_cluster_points = 100000
    request.voxelize_cloud = True
    request.leaf_size = 0.01
    #request.sensor_topic_name = "/front_xtion/depth_registered/points"

    begin_time = time.time()

    try:
        result_rois = get_object_roi_client.call(request)
    except:
        print "Something failed when trying to get ROIs"
    
    
    print "Time: {0}".format(time.time() - begin_time)
    print result_rois

    image_msg = rospy.wait_for_message(image_topic, Image)
    image = bridge.imgmsg_to_cv2(image_msg, "bgr8")

    for roi in result_rois.object_rois:
        center_x = (roi.right + roi.left) / 2
        center_y = (roi.bottom + roi.top) / 2

        left_right_padding = roi.right - roi.left
        top_bottom_padding = roi.bottom - roi.top
        padding = left_right_padding if left_right_padding > top_bottom_padding else top_bottom_padding
        padding /= 2
        padding += 10
        height, width, _ = image.shape

        left = center_x - padding if center_x - padding > 0 else 0
        right = center_x + padding if center_x + padding < width else width
        top = center_y - padding if center_y - padding > 0 else 0
        bottom = center_y + padding if center_y + padding < height else height

        cropped_image = image[top:bottom, left:right]
        cv2.imshow("Cropped Image", cropped_image)
        cv2.waitKey(0)
        
        cv2.imwrite(DATADIR + str(imageIdx) + '.jpg', image)
        coords = np.zeros(4)
        coords[0] = top
        coords[1] = bottom
        coords[2] = left
        coords[3] = right
        np.save(DATADIR + str(imageIdx), coords)
        imageIdx += 1