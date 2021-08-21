import rospy 
from bounding_box_server.srv import GetBoundingBox, GetBoundingBoxRequest

rospy.init_node("testing")
bounding_box_client = rospy.ServiceProxy("/get_bounding_boxes", GetBoundingBox)

print "Waiting for get_bounding_box server"
bounding_box_client.wait_for_service()
print "Connected"

goal_msg = GetBoundingBoxRequest()
goal_msg.transform_to = "new_frame_id"
goal_msg.cluster_threshold = 0.05
goal_msg.surface_threshold = 0.015
goal_msg.min_x = 0.05
goal_msg.max_x = 0.8

try:
  result = bounding_box_client(goal_msg)
  print result
  	 
 
except:
  print "Something went wrong in the bounding box server, see the bounding_box_server.launch terminal output"
