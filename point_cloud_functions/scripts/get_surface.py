import rospy 
from point_cloud_functions.srv import GetSurface, GetSurfaceRequest 

rospy.init_node("get_surface_test")
get_surface_client = rospy.ServiceProxy("/get_surface", GetSurface)

get_surface_msg = GetSurfaceRequest()
get_surface_msg.surface_threshold = 0.015
get_surface_msg.transform_to = "root"
get_surface_msg.min_z = -0.05
get_surface_msg.max_z = 0.05

try:
  result = get_surface_client(get_surface_msg)
except Exception as e:
  print e
  exit()

print result



