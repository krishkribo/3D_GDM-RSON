""" data extraction """

import numpy as np 
import cv2
from tqdm import tqdm
from PIL import Image
#
import os
import sys
from time import time
import time

import tensorflow as tf 
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# To fix the "No Algorithm worked!" error
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class model_preperation(object):

    def load_pre_trained(self):
        vgg_net = VGG16(include_top=True,weights='imagenet')
        vgg_input = vgg_net.input
        output = vgg_net.get_layer('fc1').output
        model = tf.keras.Model(inputs=vgg_input,outputs=output)

        return model

class processing(model_preperation):
    
    def __init__(self):
        self.dataset_path = "../../DMRSON_dataset_files/dataset/core50_128x128/core50_128x128/"
        self.data_folder = "data"
        self.working_dir = os.path.abspath(os.path.join(''))
        self.no_of_categories = int(10/2)

        if self.working_dir not in sys.path:
            sys.path.append(self.working_dir)
        #print(os.listdir())
        if self.data_folder not in os.listdir():
            os.mkdir(self.working_dir+'/'+self.data_folder)
        
        self.data = []
        self.objects = []
        self.obj_labels = ['plug_adapters','mobile_phones','scissors','light_bulbs','cans','glasses','balls','markers','cups','remote_controls']

    def write_file(self,file,value,path):
        # save the numpy array 
        if not file in os.listdir(path):
            with open(self.working_dir+'/'+file,'wb') as f:
                np.save(f,np.array(value))
                print(f"Features stored at : {file} ")

    def load_file(self,file):
        # load features 
        with open(file,'rb') as f:
            value = np.load(f,allow_pickle=True)
            print("Features readed")

        return value    

    def get_data(self):
        for s in tqdm(os.listdir(self.dataset_path)):
            #if s not in self.non_training_scenes:
            for d in tqdm(os.listdir(self.dataset_path+'/'+s)):
                self.objects.append(d)
                for img in os.listdir(self.dataset_path+'/'+s+'/'+d):
                    #images = cv2.resize(cv2.imread(dataset_path+'/'+s+'/'+d+'/'+img),(244,244)).astype(np.float32)
                    images = Image.open(self.dataset_path+'/'+s+'/'+d+'/'+img).resize((224,224))
                    images = np.asarray(images)
                    e_labels = self.objects.index(d) # instance level labels
                    s_labels = self.obj_labels.index(self.obj_labels[int(e_labels/self.no_of_categories)]) # category level labels
                    self.data.append([images,e_labels,s_labels])
            self.objects = []
            
            time.sleep(0.5) # sleep for 5 secs
            self.write_file(self.data_folder+'/'+s+'_data.npy',self.data,self.data_folder)
            self.data = []

    def get_features(self):
        self.features_folder = "features"
        # load CNN model
        model = super().load_pre_trained()
        for s in os.listdir(self.data_folder):
            scene_no = s.split('_')[0]
            # load data and labels 
            data = self.load_file(self.data_folder+'/'+s)
            features_predicted = [np.array([model.predict(preprocess_input(d[0].reshape((1,d[0].shape[0],d[0].shape[1],d[0].shape[2]))))
                ,d[1],d[2]],dtype=object) for d in tqdm(data)]
            self.write_file(self.features_folder+'/'+scene_no+'_feature.npy',features_predicted,self.features_folder)



if __name__ == "__main__":
    processing = processing()
    dataset = processing.get_features()
    #data = processing.load_file("features/s7_feature.npy")
    #print(data[0][0].shape)

