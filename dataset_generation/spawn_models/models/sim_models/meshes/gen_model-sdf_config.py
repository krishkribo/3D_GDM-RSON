"""
Custom sdf and config file genration
"""

import os
import sys
from copy import deepcopy
import numpy as np

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
            <uri>model://modelname.dae</uri>
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
            <uri>model://modelname.dae</uri>
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
config_model = """<?xml version="1.0"?>
<model>
    <name>m_name</name>
    <version>1.0</version>
    <sdf version="1.5">m_name.sdf</sdf>

    <author>
        <name>username</name>
        <email>email</email>
    </author>

    <description>
        A m_name model.
    </description>
</model>
"""

 # add current folder to the system path
sys_path = os.path.abspath(os.path.join(".."))
if sys_path not in sys.path:
    sys.path.append(sys_path)

model_dir = "" # model directory

def get_value() -> (dict,list,list,list):
    models_dict = {}
    scene_dir = []
    scale_dir = [] 
    # add the folder containing the .dae file along with it's scale folders 
    #print(os.path.abspath(model_dir))
    for m_dir in os.listdir(os.path.abspath(model_dir)):
        if os.path.isdir(m_dir):
            if m_dir[0] == 's':
                for s_dir in os.listdir(os.path.abspath(m_dir+'/')): 
                    if s_dir.split('_')[0] == 'scale':
                        scene_dir.append(m_dir)
                        scale_dir.append(s_dir)
                        models_dict[m_dir+'_'+str(float(s_dir.split('_')[-1])/100)] = os.listdir(
                            os.path.abspath(m_dir+'/'+s_dir))
    #print(models_dict)
    # get all the model names 
    models = []
    for (_,value),s_data,c_data in zip(models_dict.items(),scene_dir,scale_dir):
        for m in value:
            if not os.path.isdir(s_data+'/'+c_data+'/'+m):
                models.append(m.split('.')[0])
    #print(models)
    # get the unique models to set the model index for data annotation
    unique_model = list(np.unique(np.array(models)))
    #print(unique_model)

    return models_dict, scene_dir, scale_dir, unique_model

def del_all_sdf():
    try:
        # note this command assumes that current folder contains the structure scene_dir/scale_dir/..
        os.system("find . -name \*.sdf -type f -delete")
        print(f"All sdf files deleted")
    except Exception as e:
        print(f"Error : {e}")

def del_all_config():
    try:
        # note this command assumes that current folder contains the structure scene_dir/scale_dir/..
        os.system("find . -name \*.config -type f -delete")
        print(f"All config filed deleted")
    except Exception as e:
        print(f"Error : {e}")

def run():
    models_dict, scene_dirs, scale_dirs, unique_model = get_value()
    print(models_dict)
    for (scene_scale,model_name),scene_dir,scale_dir in zip(models_dict.items(),scene_dirs,scale_dirs):
        for model in model_name:
            scale = float(scene_scale.split('_')[-1])
            if os.path.isfile(scene_dir+'/'+scale_dir+'/'+model):
                model = model.split('.')[0]
                model_index = unique_model.index(model)
                # sdf file generation
                sdf_model_cpy = deepcopy(sdf_model)
                sdf_model_cpy = sdf_model_cpy.replace('modelname',model)
                sdf_model_cpy = sdf_model_cpy.replace('scale_size',str(scale))

                config_model_cpy = deepcopy(config_model)
                config_model_cpy = config_model_cpy.replace('m_name',model)
                #@Todo save the files in the req folder
                sdf_file = model+'.sdf'
                config_file = model+'.config'
                # write sdf
                if sdf_file not in os.listdir(scene_dir+'/'+scale_dir+'/'):
                    with open(scene_dir+'/'+scale_dir+'/'+sdf_file,'w') as f1:
                        f1.write(sdf_model_cpy)
                        print(f"{model} sdf file writted to the disk at location --> {scene_dir+'/'+scale_dir+'/'+sdf_file}")
                # write config 
                if config_file not in os.listdir(scene_dir+'/'+scale_dir+'/'):
                    with open(scene_dir+'/'+scale_dir+'/'+config_file,'w') as f2:
                        f2.write(config_model_cpy)
                        print(f"{model} config file writted to the disk at location --> {scene_dir+'/'+scale_dir+'/'+config_file}")
        

if __name__ == "__main__":
    import argparse

    get_args = argparse.ArgumentParser(description="selected mode of operation| 1. run| 2. delete all sdf files| 3. delete all config files")
    get_args.add_argument('--type', type=int, default=1, help="Enter the mode of operation")
    args = get_args.parse_args().type

    if args == 1:
        run()
    elif args == 2:
        del_all_sdf()
    elif args == 3:
        del_all_config()
    else:
        print(f"Enter the number 1,2 or 3")
        exit()

