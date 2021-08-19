#! /usr/bin/env python3

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import math
from geometry_msgs.msg import Point

class draw_marker(object):

    def __init__(self):
        
        self.publisher = rospy.Publisher('visual_marker',Marker,queue_size=1)
        self.marker_array = MarkerArray()

        self.marker = Marker()
        self.marker.header.frame_id = 'new_frame_id'
        self.marker.type = self.marker.LINE_STRIP
        self.marker.action = self.marker.ADD
        self.marker.scale.x = 0.01
        self.marker.scale.y = 0.01
        self.marker.scale.z = 0.01
        self.marker.color.a = 1.0 
        self.marker.color.r = 0.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0

        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0
        self.marker.pose.position.x = 0.0
        self.marker.pose.position.y = 0.0
        self.marker.pose.position.z = 0.12

        self.marker.points = []
        # start point
        first_line_point = Point()
        first_line_point.x = -0.3475
        first_line_point.y = -0.3759
        first_line_point.z = 0.0
        self.marker.points.append(first_line_point)
        # second point
        second_line_point = Point()
        second_line_point.x = 0.3475
        second_line_point.y = -0.3759
        second_line_point.z = 0.0
        self.marker.points.append(second_line_point) 
        # thrid point
        thrid_line_point = Point()
        thrid_line_point.x = 0.75
        thrid_line_point.y = 0.8
        thrid_line_point.z = -0.18
        self.marker.points.append(thrid_line_point) 
        # fourth point
        fourth_line_point = Point()
        fourth_line_point.x = -0.75
        fourth_line_point.y = 0.8
        fourth_line_point.z = -0.18
        self.marker.points.append(fourth_line_point) 
        # fifth point
        fifth_line_point = Point()
        fifth_line_point.x = -0.3475
        fifth_line_point.y = -0.3759   
        fifth_line_point.z = 0.0
        self.marker.points.append(fifth_line_point) 

        self.publisher.publish(self.marker)



if __name__ == "__main__":
    rospy.init_node("marker_topic")
    while not rospy.is_shutdown():
        draw_marker()



