#include "point_cloud_functions/point_cloud_functions.h"

PointCloudFunctions::PointCloudFunctions(ros::NodeHandle &nh) : 
    nh_(nh) {
  get_object_roi_service_ = nh_.advertiseService("/get_object_roi", &PointCloudFunctions::getObjectROICallback, this);
  voxelize_point_cloud_service_ = nh_.advertiseService("/voxelize_point_cloud", &PointCloudFunctions::voxelizePointCloudCallback, this);
  find_table_service_  = nh_.advertiseService("/find_table", &PointCloudFunctions::findTableCallback, this);
  get_closest_point_service_ = nh_.advertiseService("/get_closest_point", &PointCloudFunctions::getClosestPointCallback, this);
  get_center_objects_service_ = nh_.advertiseService("/get_center_objects", &PointCloudFunctions::getCenterObjectsCallback, this);
  get_planar_coefficients_service_ = nh_.advertiseService("/get_planar_coefficients", &PointCloudFunctions::getPlanarCoefficientsCallback, this);
  get_surface_service_ = nh_.advertiseService("/get_surface", &PointCloudFunctions::getSurfaceCallback, this);
  point_cloud_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>("/point_cloud_functions/cloud", 1);
  
  ros::param::param(std::string("~sensor_topic"), sensor_topic_name_, std::string("/camera/depth/points"));
}
