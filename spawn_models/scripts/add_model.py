#! /usr/bin/python3
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, SpawnModelResponse, DeleteModel, SetModelState, GetModelState
from copy import deepcopy
from tf.transformations import quaternion_from_euler
from time import time
import numpy as np
import time

sdf_model = """<?xml version="1.0" ?>
<sdf version="1.5">
  <model name="modelname">
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
            <scale>1.0 1.0 1.0</scale>
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
            <scale>1.0 1.0 1.0</scale>
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


def create_model_request(modelname, px, py, pz, rr, rp, ry):
    """Create a SpawnModelRequest with the parameters of the cube given.
    modelname: name of the model for gazebo
    px py pz: position of the cube (and it's collision cube)
    rr rp ry: rotation (roll, pitch, yaw) of the model
    sx sy sz: size of the cube"""
    model = deepcopy(sdf_model)
    # Replace position
    pose_model = str(px) + " " + \
        str(py) + " " + str(pz)
    #  + " " + str(rr) + " " + str(rp) + " " + str(ry)
    model = model.replace('POSE', pose_model)
    # Replace modelname
    model = model.replace('modelname', str(modelname))

    req = SpawnModelRequest()
    req.model_name = modelname
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

def delete_model_request(model_name):
    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    delete_model(model_name)


def set_model_request(modelname,x,y,z,wx,wy,wz,ww):
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

if __name__ == '__main__':
    rospy.init_node('object_functions')
    # service call requests 
    spawn_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    get_state = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
    
    rospy.loginfo("Waiting for /gazebo/spawn_sdf_model service...")
    spawn_srv.wait_for_service()
    rospy.loginfo("Connected to service!")
    
    rospy.loginfo("Waiting for /gazebo/set_model service...")
    set_state.wait_for_service()
    rospy.loginfo("Connected to service!")

    rospy.loginfo("Waiting for /gazebo/get_model service...")
    get_state.wait_for_service()
    rospy.loginfo("Connected to service!")


    models = ["pope","bottle","bowl","Heniken_can","coke_can"]
    # delete_model 
    delete_model_request(models[1])
    # Spawn object 1
    rospy.loginfo("Spawning model")
    req1 = create_model_request(models[1],
                              0.0, 0.0, 0.003076,  # position
                              0.0, 0.0, 0.0,  # rotation
                              )  # size
    spawn_srv.call(req1)
    print("waiting for timeout")
    rospy.sleep(1.0)
    print ("set state called")

    for x in np.linspace(-0.5, 0.3, 100):
      try:
        object_state = get_state(models[1],'world')
        assert object_state.success is True
      except rospy.ServiceException:
        print ("get_state service failed")

      print(object_state.pose)

      set_state.call(set_model_request(models[1],
                                0.0, 0.0, 0.003076,  # position
                                0.0, 0.0, x,  # rotation
                                1.0))
      #spawn_srv.call(req1)
      time.sleep(0.5)
