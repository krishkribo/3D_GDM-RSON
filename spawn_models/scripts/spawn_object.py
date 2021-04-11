#!/usr/bin/env python
import rospy 
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import *
import tf
import rospkg
from math import pi
from random import shuffle, uniform, randint


def spawn_object(object_name, x, y, z, rotation):
    path = rospack.get_path("spawn_models")
    with open(path +  "/models/" + str(object_name) + ".sdf") as f:
        box_xml = f.read()
        
    orientation = Quaternion()
    quaternion = tf.transformations.quaternion_from_euler(0, 0, rotation)
    orientation.x = quaternion[0]
    orientation.y = quaternion[1]
    orientation.z = quaternion[2]
    orientation.w = quaternion[3]
    
    pose = Pose(Point(x, y, z), orientation)
    
    spawn_model(object_name, box_xml, "", pose, "world")            
        
if __name__ == "__main__":
    rospy.init_node("set_objects")
    rospack = rospkg.RosPack()
    print 'Waiting for service...'
    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    print 'Connected to service'
    
    spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    
    #objects = ["box0", "box1", "box2", "box3", "box4"]
    
    #for obj in objects:
    delete_model("model")

    place = [0.1, 0] 
    spawn_object("model", place[0], place[1], 0.0, 0)