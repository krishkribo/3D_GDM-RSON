#include "bounding_box_server/bounding_box_server.h"

//! The main function to initialize the ros node and create the BoundingBoxServer object
int main(int argc, char **argv) {
  ros::init(argc, argv, "bounding_box_server_node");
  ros::NodeHandle nh;
  BoundingBoxServer bounding_box_server(nh);

  ros::spin();
}