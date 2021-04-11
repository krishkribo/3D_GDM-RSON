#! /usr/bin/env python3
from get_ROI import get_object_roi
from image_converter import ImageConverter
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
import rospy

class ObjectDetection(object):
    """
    Generated the bounding boxes for objects from the sensor images
    """
    def __init__(self):
        self.image_converter = ImageConverter() # For converting from sensor_msgs/Image to OpenCv Image
        self.image_publisher = rospy.Publisher("/object_image", Image, queue_size=1) # Publisher for publising image
        self.roi_service = get_object_roi()

    def run(self):
        while not self.roi_service.flag == True:
            try:
                img_msg = rospy.wait_for_message("/camera/color/image_raw",Image,rospy.Duration(1.0))
                # get images and roi from 
                roi_image = self.image_converter.convert_to_opencv(img_msg)
                roi_objects= self.roi_service.get_roi()
                image = roi_image.copy()
                
                for roi in roi_objects:
                    # draw the predicted labels and rectangles on the image to publish 
                    cv2.rectangle(image,(roi.left, roi.top), (roi.right, roi.bottom), (255, 0, 0))
                    
                    # covert to image to sensor_msgs\Image format and publish image to image publisher
                    image_msg = self.image_converter.convert_to_ros(image,"bgr8")
                    image_msg.header.frame_id = "new_frame_id"
                    self.image_publisher.publish(image_msg)

            except Exception as e:
                pass

if __name__ == "__main__":
    rospy.init_node("Object_detection")
    object_detection = ObjectDetection()
    object_detection.run()
    print('process done')
    exit()
    #rospy.spin()

