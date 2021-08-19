# Lifelong Object Recognition and Grasp Synthesis using Dual Memory Recurrent Self-Organization Networks

This repository contains the implementation of synthetic sequential point cloud dataset generation and the implementation of hybrid learning architecture, comparises of generative autoencoder and the growing dual memory networks (GDM) for contiual learning of object recognition and grasping synthesis towards lifelong learning fashion.

 * ``` dataset_generation ``` folder contains the implementation of sequential point cloud dataset generation in _ROS melodic_. 
 * ``` CNN ``` folder contains the implementation and evaluation of the proposed autoencoder model. 
 * ``` GDM_learning ``` as the name implies, conatins the implementation and evaluation of the updated and modified version of growing dual-memory networks in bacth and incremental learning scenarios. 
 * ``` Grasping and dual_pipeline.py ``` contains the end-end implementation of the proposed system architecture in _pybullet_ simulation environment for pick and place, and pack manipulation tasks.

## Requirements and Installation
* python 3.8
* Install the required libraries by running ``` pip install -r requirements.txt```
The ```requirements.txt ``` file contains the esssential libraries for both dataset generation and continual learning. 

## 

 


Thesis work, University of Groningen
