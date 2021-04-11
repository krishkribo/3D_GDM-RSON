import os 
import sys
import numpy as np
from tqdm import tqdm
path = os.path.abspath(os.path.join('..'))
if path not in sys.path:
    sys.path.append(path)

print(os.getcwd())
data_dir = 'src/spawn_models/models/sim_models/meshes'

data_list = os.listdir(os.path.abspath(data_dir))
scale_list = []
scene_list = []
model_dict = {}
for data in data_list:
    #print(data)
    if os.path.isdir(data_dir+'/'+data):
        if data[0] == 's':
            for scale_data in os.listdir(data_dir+'/'+data+'/'):
                if scale_data.split('_')[0] == 'scale':
                    scene_list.append(data)
                    scale_list.append(scale_data)
                    #model_list.append(os.listdir(os.path.abspath(data_dir+'/'+data)))
                    model_dict[data+'_'+str(float(scale_data.split('_')[-1])/100)] = os.listdir(
                        os.path.abspath(data_dir+'/'+data+'/'+scale_data))

scale_list = np.array(scale_list)
#print(scale_list)
#print(scene_list)
#print(model_dict)
test_list = []
models = []
for (_,value),s_data,c_data in tqdm(zip(model_dict.items(),scene_list,scale_list),desc='main bar'):
    for m in tqdm(value,desc='scale bar'):
        if not os.path.isdir(data_dir+'/'+s_data+'/'+c_data+'/'+m):
            models.append(m.split('.')[0])
print(models)
unique_model = list(np.unique(np.array(models)))
print(unique_model)
print([unique_model.index(m) for m in models])

#for s in scene_list:
"""for (k,v),s,c in zip(model_dict.items(),scene_list,scale_list):
    #print(c)
    #print(k)
    print(float(k.split('_')[-1]))
    for model in v:
        if os.path.isfile(data_dir+'/'+s+'/'+c+'/'+model):
            print(model)
            #print(model.split('.')[0])
            #print(os.path.abspath(model))
            #print('/'+c+'/'+model)
    print("-----")
"""