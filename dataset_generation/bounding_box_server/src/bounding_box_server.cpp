#include "bounding_box_server/bounding_box_server.h"

//! Constructor.
/** Get the point_cloud_topic giving via a launch files, otherwise uses default "/front_xtion/depth_registered/points".
 * Creates the services and the timer for running the bounding box server.
 * The timer is disabled by default and needs to be enabled via a service call.
 * transform_to_link_ is hardcoded ro "/root", this will also be the arms planning frame of reference
 * \param nh a ros::NodeHandle created in the main function.  
 */ 
BoundingBoxServer::BoundingBoxServer(ros::NodeHandle &nh) :
    nh_(nh),
    bounding_box_publisher_(nh_) {

  ros::param::param(std::string("~point_cloud_topic"), point_cloud_topic_, std::string("/camera/depth/points"));
  
  get_bounding_boxes_service_ = nh_.advertiseService("/get_bounding_boxes", &BoundingBoxServer::getBoundingBoxesCallback, this);
}
