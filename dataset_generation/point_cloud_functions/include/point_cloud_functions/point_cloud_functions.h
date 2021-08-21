#ifndef _H_POINT_CLOUD_FUNCTIONS__
#define _H_POINT_CLOUD_FUNCTIONS__

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>

#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <point_cloud_functions/GetObjectROI.h>
#include <point_cloud_functions/VoxelizePointCloud.h>
#include <point_cloud_functions/FindTable.h>
#include <point_cloud_functions/GetClosestPoint.h>
#include <point_cloud_functions/GetCenterObjects.h>
#include <point_cloud_functions/GetPlanarCoefficients.h>
#include <point_cloud_functions/GetSurface.h>

#include <vector>

typedef pcl::PointXYZRGB Point;
typedef pcl::PointCloud<Point> PointCloud;
typedef pcl::PointCloud<Point>::Ptr PointCloudPtr;

struct FilterOptions {
  std::string transform_to_link;
  bool filter_x = false;
  bool filter_y = false;
  bool filter_z = false;
  float min_x, min_y, min_z;
  float max_x, max_y, max_z;
};

class PointCloudFunctions {

  ros::NodeHandle nh_;
  ros::ServiceServer get_object_roi_service_;
  ros::ServiceServer voxelize_point_cloud_service_;
  ros::ServiceServer find_table_service_;
  ros::ServiceServer get_closest_point_service_;
  ros::ServiceServer get_center_objects_service_;
  ros::ServiceServer get_planar_coefficients_service_;
  ros::ServiceServer get_surface_service_;
  tf::TransformListener transform_listener_;

  ros::Publisher point_cloud_publisher_;

  pcl::PassThrough<Point> pass_through_filter_;
  pcl::SACSegmentation<Point> sac_segmentation_;
  pcl::EuclideanClusterExtraction<Point> cluster_extraction_;
  pcl::VoxelGrid<Point> voxel_grid_;
  pcl::ExtractIndices<Point> extract_indices_;

  std::string sensor_topic_name_;

  public: 
    PointCloudFunctions(ros::NodeHandle &nh);

  private: 
    bool getPointCloud(PointCloudPtr point_cloud, point_cloud_functions::GetObjectROIRequest &req);
    bool getPointCloud(PointCloudPtr point_cloud, point_cloud_functions::FindTableRequest &req);
    bool getPointCloud(PointCloudPtr point_cloud, point_cloud_functions::GetClosestPointRequest &req);
    bool getPointCloud(PointCloudPtr point_cloud, point_cloud_functions::GetCenterObjectsRequest &req);
    bool getPointCloud(PointCloudPtr point_cloud, point_cloud_functions::GetSurfaceRequest &req);
    bool getPointCloud(PointCloudPtr point_cloud, FilterOptions filter_options);
    bool getPointCloud(PointCloudPtr original_point_cloud, PointCloudPtr modified_point_cloud, point_cloud_functions::GetObjectROIRequest &req);
    bool getPointCloud(PointCloudPtr point_cloud, point_cloud_functions::GetPlanarCoefficientsRequest &req);
    bool getSensorMessage(sensor_msgs::PointCloud2 &point_cloud_msg, std::string sensor_topic_name);
    bool transformPointCloudTo(sensor_msgs::PointCloud2 &point_cloud_msg, sensor_msgs::PointCloud2 &transformed_point_cloud_msg, std::string transform_to_link);
    void getPointCloudFromSensorMessage(sensor_msgs::PointCloud2 &point_cloud_msg, PointCloudPtr point_cloud);
    void getSensorMessageFromPointCloud(PointCloudPtr point_cloud, sensor_msgs::PointCloud2 &point_cloud_msg);

    void passThroughFilter(PointCloudPtr point_cloud_in, PointCloudPtr point_cloud_out, std::string field_name, float min_value, float max_value, bool filter_negatives = false);
    void voxelizePointCloud(PointCloudPtr point_cloud, PointCloudPtr voxelized_point_cloud, float leaf_size);
    void segmentPointCloud(PointCloudPtr point_cloud, PointCloudPtr segmented_point_cloud, bool remove_plane, float distance_threshold);
    void getClusters(PointCloudPtr point_cloud, std::vector<PointCloudPtr> &cluster_clouds, size_t min_cluster_points, size_t max_cluster_points, float cluster_distance);
    void getPlanarCoefficients(PointCloudPtr point_cloud, float surface_threshold, pcl::ModelCoefficients::Ptr coefficients);

    bool getObjectROICallback(point_cloud_functions::GetObjectROIRequest &req, point_cloud_functions::GetObjectROIResponse &res);
    bool voxelizePointCloudCallback(point_cloud_functions::VoxelizePointCloudRequest &req, point_cloud_functions::VoxelizePointCloudResponse &res);
    bool findTableCallback(point_cloud_functions::FindTableRequest &req, point_cloud_functions::FindTableResponse &res);
    bool getClosestPointCallback(point_cloud_functions::GetClosestPointRequest &req, point_cloud_functions::GetClosestPointResponse &res);
    bool getCenterObjectsCallback(point_cloud_functions::GetCenterObjectsRequest &req, point_cloud_functions::GetCenterObjectsResponse &res);
    bool getPlanarCoefficientsCallback(point_cloud_functions::GetPlanarCoefficientsRequest &req, point_cloud_functions::GetPlanarCoefficientsResponse &res);
    bool getSurfaceCallback(point_cloud_functions::GetSurfaceRequest &req, point_cloud_functions::GetSurfaceResponse &res);

    point_cloud_functions::ObjectROI getROI(PointCloudPtr original_point_cloud, PointCloudPtr cluster_cloud);

    Point findClosestPoint(PointCloudPtr point_cloud);
    Point findClosestPoint(PointCloudPtr point_cloud, Point search_point, int K = 10);

    bool transformPointCloud (const std::string &target_frame, const sensor_msgs::PointCloud2 &in, 
                     sensor_msgs::PointCloud2 &out, const tf::TransformListener &tf_listener);
    void transformAsMatrix (const tf::Transform& bt, Eigen::Matrix4f &out_mat);
    void transformPointCloud (const Eigen::Matrix4f &transform, const sensor_msgs::PointCloud2 &in,
                     sensor_msgs::PointCloud2 &out);
};

#endif
