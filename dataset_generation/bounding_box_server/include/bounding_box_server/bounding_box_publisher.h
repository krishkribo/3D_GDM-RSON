#ifndef _H_BOUNDING_BOX_PUBLISHER__
#define _H_BOUNDING_BOX_PUBLISHER__

#include <ros/ros.h>
#include <bounding_box_server/BoundingBox.h>
#include <bounding_box_server/BoundingBoxes.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <vector>
#include <tf/tf.h>


class BoundingBoxPublisher {
  
  ros::NodeHandle nh_;
  ros::Publisher bounding_box_publisher_;
  ros::Publisher bounding_box_marker_array_publisher_;
  std::vector<std::array<float, 6>> offsets_;
  public:
    BoundingBoxPublisher(ros::NodeHandle &nh);
    void publishBoundingBoxes(std::vector<bounding_box_server::BoundingBox> bounding_boxes, std::string transform_to);
};

#endif 