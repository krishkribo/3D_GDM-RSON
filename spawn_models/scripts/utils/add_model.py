#! /usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModelRequest
from copy import deepcopy
from tf.transformations import quaternion_from_euler
from time import time
import numpy as np
import time

class generate_model(object):
    """
    Add the object models to the gazebo world and 
    perform the object movement actions in the world
    """
    def __init__(self):
        self.sdf_model = """<?xml version="1.0" ?>
        <sdf version="1.5">
        <model name="m_name">
        <static>0</static>
            <link name="model_link">
            <inertial>
                <pose>0 0 0.00523 0 0 0</pose>
                <mass>0.1</mass>
                <inertia>
                <ixx>0.01</ixx>
                <ixy>0</ixy>
                <ixz>0</ixz>
                <iyy>0.01</iyy>
                <iyz>0</iyz>
                <izz>0.01</izz>
                </inertia>
            </inertial>
            <collision name="collision">
                <pose>0 0 0 0 0 0</pose>
                <geometry>
                <mesh>
                    <uri>model://sim_models/meshes/modelname.dae</uri>
                    <scale>scale_size scale_size scale_size</scale>
                </mesh>
                </geometry>
                <surface>
                <bounce />
                <friction>
                    <ode>
                    <mu>1.0</mu>
                    <mu2>1.0</mu2>
                    </ode>
                </friction>
                <contact>
                    <ode>
                    <kp>10000000.0</kp>
                    <kd>1.0</kd>
                    <min_depth>0.0</min_depth>
                    <max_vel>0.0</max_vel>
                    </ode>
                </contact>
                </surface>

            </collision>
            <visual name="visual">
                <pose>0 0 0 0 0 0</pose>
                <geometry>
                <mesh>
                    <uri>model://sim_models/meshes/modelname.dae</uri>
                    <scale>scale_size scale_size scale_size</scale>
                </mesh>
                </geometry>
            </visual>
            
            <velocity_decay>
                <linear>0.000000</linear>
                <angular>0.000000</angular>
            </velocity_decay>
            <self_collide>0</self_collide>
            <kinematic>0</kinematic>
            <gravity>0</gravity>

            </link>
        </model>
        </sdf>
        """


    def create_model_request(self,modelname,scale, px, py, pz, rr, rp, ry):
        """ Create model request """
        #print(modelname)
        #print(modelname.split('/')[-1])
        model = deepcopy(self.sdf_model)
        # Replace position
        pose_model = str(px) + " " + \
            str(py) + " " + str(pz)
        #  + " " + str(rr) + " " + str(rp) + " " + str(ry)
        model = model.replace('POSE', pose_model)
        # Replace modelname
        model = model.replace('modelname', str(modelname))
        model = model.replace('m_name',modelname.split('/')[-1])
        model = model.replace('scale_size',str(scale))
        req = SpawnModelRequest()
        req.model_name = modelname.split('/')[-1]
        req.model_xml = model
        req.initial_pose.position.x = px
        req.initial_pose.position.y = py
        req.initial_pose.position.z = pz

        q = quaternion_from_euler(rr, rp, ry)
        req.initial_pose.orientation.x = q[0]
        req.initial_pose.orientation.y = q[1]
        req.initial_pose.orientation.z = q[2]
        req.initial_pose.orientation.w = q[3]

        return req

    def delete_model_request(self,model_name,delete_model):
        """ Delete model request """
        #print(model_name)
        delete_model(model_name)


    def set_model_request(self,modelname,x,y,z,wx,wy,wz,ww):
        """ set model request """
        state_msg = ModelState()
        state_msg.model_name = modelname
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = z
        state_msg.pose.orientation.x = wx
        state_msg.pose.orientation.y = wy
        state_msg.pose.orientation.z = wz
        state_msg.pose.orientation.w = ww

        return state_msg
