# Lifelong Object Grasp Synthesis using Dual Memory Recurrent Self-Organization Networks

## Sequential 3D-Dataset Generation branch 
### Installation Note: 
For Ros data generation process need to install python3 tf2, follow in the instructions in the link below to install python3 tf2

https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/

### Instructions to run the code 
#### Initialization:
* start roscore ```roscore```
* launch gazebo ```roslaunch sim_world simulation.launch```
#### Start the bounding box server
* ```roslaunch bounding_box_server bounding_box_server.launch```
#### Run the object detection script in spawn_models/scripts folder
* ```python3 utils/object_detection.py```
#### Run the data generation script in spawn models/scripts folder
* ```python3 generate_data.py```

Use the two rviz config files ```rviz_window1.rviz and rviz_window2.rviz``` to visualize environment, boudingbox, object detection and point clouds of the objects used in the data generation process.  

### Model Note:
* The models provided in the spawn_models/models/sim_models/meshes folder is for sample only. The complete model data used in the data generation process are 
available at https://drive.google.com/drive/folders/1O0PbzgXksPJ_bQMj5TcLGX566lPRlv8R?usp=sharing (s1-s5)

* ```gen_model-sdf_config.py``` file in spawn_models folder generates the custom sdf and config for all models. 

* The models with custom sdf and config files used in this experiment are available at https://drive.google.com/drive/folders/1216AoY10cXdcDJ8BrWmL3o3oIvzx1j7m?usp=sharing which contain 25 object categories with 5 objects in each category.

Thesis work, University of Groningen

