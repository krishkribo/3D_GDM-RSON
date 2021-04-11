#include <ros/ros.h>
#include "point_cloud_functions/point_cloud_functions.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "point_cloud_functions_node");
  ros::NodeHandle nh;
  PointCloudFunctions point_cloud_functions(nh);
  ros::spin();
}