#include <ros/ros.h>
#include <point_cloud_functions/VoxelizePointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <chrono>

ros::ServiceClient voxelize_client_;
ros::Publisher point_cloud_publisher_;
ros::Subscriber point_cloud_subscriber_;

std::string input_topic_name_;
std::string output_topic_name_;
float leaf_size_;

void voxelizePointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &point_cloud_msg) {
  point_cloud_functions::VoxelizePointCloudRequest request;
  point_cloud_functions::VoxelizePointCloudResponse response;
  request.point_cloud = *point_cloud_msg;
  request.leaf_size = leaf_size_;

  if (voxelize_client_.call(request, response)) {
    point_cloud_publisher_.publish(response.voxelized_point_cloud);
  } 
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "voxelize_pointcloud");
  ros::NodeHandle nh;
  voxelize_client_ = nh.serviceClient<point_cloud_functions::VoxelizePointCloud>("/voxelize_point_cloud");
  
  ros::NodeHandle private_nh("~");
  private_nh.param(std::string("input_topic_name"), input_topic_name_, std::string("/camera/depth/points"));
  private_nh.param(std::string("output_topic_name"), output_topic_name_, std::string("/voxelized_cloud/points"));
  private_nh.param(std::string("leaf_size"), leaf_size_, float(0.01f));
  
  ROS_INFO_STREAM("Input topic name: " << input_topic_name_);
  point_cloud_publisher_ = nh.advertise<sensor_msgs::PointCloud2>(output_topic_name_, 1);
  point_cloud_subscriber_ = nh.subscribe<sensor_msgs::PointCloud2>(input_topic_name_, 1, voxelizePointCloudCallback);

  voxelize_client_.waitForExistence();

  ros::spin();
}
