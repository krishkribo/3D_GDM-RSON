import numpy as np
import cv2 


a = np.array((
    [6,2,3],
    [4,9,1],
    [7,8,5]))

print(a)

# find the min and max 
# get the min_max value of the original data 
g_min_val, g_max_val, _, _ = cv2.minMaxLoc(a)
min_idx_list = []
min_val_list = []
for i in range(9):
    min_val, max_val, min_idx, max_idx = cv2.minMaxLoc(a)
    print(f"g_min : {g_min_val}")
    print(f"min : {min_val}")
    if [min_idx[1], min_idx[0]] not in min_idx_list:
        if min_val - g_min_val >= 3 or i == 0:
            min_idx_list.append([min_idx[1], min_idx[0]])
            min_val_list.append(min_val)
            g_min_val = min_val
    
    a[min_idx[1], min_idx[0]] += g_max_val

        

print(min_idx_list)
print(min_val_list)




