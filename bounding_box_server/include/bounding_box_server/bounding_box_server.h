#ifndef _H_BOUNDING_BOX_SERVER__
#define _H_BOUNDING_BOX_SERVER__

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <actionlib/server/simple_action_server.h>

#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>

#include <Eigen/Eigen>
#include <vector>

#include <bounding_box_server/BoundingBox.h>
#include <bounding_box_server/BoundingBoxes.h>
#include <bounding_box_server/GetBoundingBox.h>
#include <std_srvs/Empty.h>

#include "bounding_box_publisher.h"

typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> PointCloud;
typedef pcl::PointCloud<Point>::Ptr PointCloudPtr; 

class BoundingBoxServer {
  //! NodeHandle for creating publishers and services
  ros::NodeHandle nh_;
  //! Point cloud topic from which to receive the Point Cloud Sensor data
  std::string point_cloud_topic_;
  //! Transform listener used for transforming the Point Cloud data to a different coordinate system
  tf::TransformListener transform_listener_;
  //! Publisher to publish the bounding boxes with the BoundingBoxes message
  BoundingBoxPublisher bounding_box_publisher_;
  //! Service to get bounding boxes data
  ros::ServiceServer get_bounding_boxes_service_;

  //! PCL Class for filtering out parts of the Point Cloud
  pcl::PassThrough<Point> pass_through_filter_;
  //! PCL Class for segmenting the Point Cloud, used for remove the table surface
  pcl::SACSegmentation<Point> sac_segmentation_;
  //! PCL Class for extracting specific points based on indices 
  pcl::ExtractIndices<Point> extract_indices_;
  //! PCL Class for extracting clusters from a Point Cloud
  pcl::EuclideanClusterExtraction<Point> cluster_extraction_;
  //! PCL Class for project a Point Cloud
  pcl::ProjectInliers<Point> project_inliers_;
  //! PCL Class for reducing the Point Cloud in based on a voxel filter
  pcl::VoxelGrid<Point> voxel_grid_; 
  //! PCL Class for performing Principal Component Analysis
  pcl::PCA<Point> pca_;
  //! Timer thats run the algorithms leading to publishing the bounding boxes data
  ros::Timer update_timer_;
  //! String that indices the link to transform the Point Cloud data to
  const std::string transform_to_link_;

  public: 
    BoundingBoxServer(ros::NodeHandle &nh);
    bool getBoundingBoxesCallback(bounding_box_server::GetBoundingBoxRequest &req, bounding_box_server::GetBoundingBoxResponse &res);

  private:
    bool getPointCloudMessage(sensor_msgs::PointCloud2 &point_cloud_message);
    bool transformPointCloudMessageToLink(sensor_msgs::PointCloud2 &point_cloud_message, sensor_msgs::PointCloud2 &transformed_point_cloud_message, std::string transform_to_link);
    void passThroughFilter(PointCloudPtr point_cloud, PointCloudPtr filtered_cloud, std::string field_name, float min_value, float max_value);
    void removeFloor(PointCloudPtr point_cloud, PointCloudPtr floorless_point_cloud, float distance_threshold);
    void voxelizePointCloud(PointCloudPtr point_cloud, PointCloudPtr voxelized_point_cloud, float leaf_size = 0.005f);
    void extractClusters(PointCloudPtr point_cloud, std::vector<PointCloudPtr> &clusters, float cluster_tolerance);
    void projectPointCloudOnPlane(PointCloudPtr point_cloud, PointCloudPtr projected_point_cloud);
    Eigen::Matrix3f getEigenVectors(PointCloudPtr point_cloud); 
    float getAngle(Eigen::Vector3f eigen_vector);
    Eigen::Vector3f getCenterPointCloud(PointCloudPtr point_cloud);
    void transformPointCloudToCenter(PointCloudPtr point_cloud, PointCloudPtr center_cloud, Eigen::Vector3f centroid_vector, float angle);
    void getDimensions(PointCloudPtr point_cloud, bounding_box_server::BoundingBox &bounding_box);
    bool transformPointCloud (const std::string &target_frame, const sensor_msgs::PointCloud2 &in, 
                     sensor_msgs::PointCloud2 &out, const tf::TransformListener &tf_listener);
    void transformAsMatrix (const tf::Transform& bt, Eigen::Matrix4f &out_mat);
    void transformPointCloud (const Eigen::Matrix4f &transform, const sensor_msgs::PointCloud2 &in,
                     sensor_msgs::PointCloud2 &out);
};

#endif
