import rospy 
from point_cloud_functions.srv import GetPlanarCoefficients, GetPlanarCoefficientsRequest

rospy.init_node("test_get_planar")

client = rospy.ServiceProxy("/get_planar_coefficients", GetPlanarCoefficients)

get_planar_coefficient = GetPlanarCoefficientsRequest()
get_planar_coefficient.transform_to = "root"
get_planar_coefficient.surface_threshold = 0.01

try:
  result = client(get_planar_coefficient)
except Exception as e:
  print "Failed", e
  exit()

print result