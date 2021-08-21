#include "point_cloud_functions/point_cloud_functions.h"

void PointCloudFunctions::passThroughFilter(PointCloudPtr point_cloud_in,
                                            PointCloudPtr point_cloud_out,
                                            std::string field_name,
                                            float min_value,
                                            float max_value,
                                            bool filter_negatives) {
  pass_through_filter_.setInputCloud(point_cloud_in);
  pass_through_filter_.setFilterFieldName(field_name);
  pass_through_filter_.setFilterLimits(min_value, max_value);
  pass_through_filter_.setNegative(filter_negatives);
  pass_through_filter_.filter(*point_cloud_out);
}                                        

void PointCloudFunctions::voxelizePointCloud(PointCloudPtr point_cloud, PointCloudPtr voxelized_point_cloud, float leaf_size) {
  voxel_grid_.setInputCloud(point_cloud); 
  voxel_grid_.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxel_grid_.filter(*voxelized_point_cloud);
}

void PointCloudFunctions::segmentPointCloud(PointCloudPtr point_cloud, PointCloudPtr segmented_point_cloud, 
                                            bool remove_plane,  float distance_threshold) {
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  
  if (distance_threshold == 0.0) {
    ROS_WARN("Distance threshold for planar segmenation is 0.0, setting value to 0.015 (1.5cm) else it will fail");
    distance_threshold = 0.015f;
  }

  sac_segmentation_.setInputCloud(point_cloud);
  sac_segmentation_.setDistanceThreshold(distance_threshold);
  sac_segmentation_.setMethodType(pcl::SAC_RANSAC);
  sac_segmentation_.setModelType(pcl::SACMODEL_PLANE);
  sac_segmentation_.setOptimizeCoefficients(true);
  sac_segmentation_.segment(*inliers, *coefficients);

  extract_indices_.setInputCloud(point_cloud);
  extract_indices_.setIndices(inliers);
  extract_indices_.setNegative(remove_plane);
  extract_indices_.filter(*segmented_point_cloud);
}

void PointCloudFunctions::getPlanarCoefficients(PointCloudPtr point_cloud, float surface_threshold, pcl::ModelCoefficients::Ptr coefficients) {
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices()); 
  
  if (surface_threshold == 0.0) {
    ROS_WARN("Surface threshold for planar segmentation is 0.0, setting the value to 0.015 (1.5cm) else it will fail");
    surface_threshold = 0.015f;
  }
  
  sac_segmentation_.setInputCloud(point_cloud);
  sac_segmentation_.setDistanceThreshold(surface_threshold);
  sac_segmentation_.setModelType(pcl::SACMODEL_PLANE);
  sac_segmentation_.setMethodType(pcl::SAC_RANSAC);
  sac_segmentation_.setOptimizeCoefficients(true);
  sac_segmentation_.segment(*inliers, *coefficients);
}

void PointCloudFunctions::getClusters(PointCloudPtr point_cloud, std::vector<PointCloudPtr> &cluster_clouds, 
                                      size_t min_cluster_points, size_t max_cluster_points, float cluster_distance) {
  pcl::search::KdTree<Point>::Ptr search_tree(new pcl::search::KdTree<Point>());
  search_tree->setInputCloud(point_cloud);

  std::vector<pcl::PointIndices> cluster_indices;

  cluster_extraction_.setInputCloud(point_cloud);
  cluster_extraction_.setMinClusterSize(min_cluster_points);
  cluster_extraction_.setMaxClusterSize(max_cluster_points);
  cluster_extraction_.setClusterTolerance(cluster_distance);
  cluster_extraction_.setSearchMethod(search_tree);
  cluster_extraction_.extract(cluster_indices);

  //ROS_INFO_STREAM("Clusters found: " << cluster_indices.size());

  for (size_t cluster_nr = 0; cluster_nr < cluster_indices.size(); ++cluster_nr) {
    PointCloudPtr cluster_cloud(new PointCloud());
    cluster_cloud->points.resize(cluster_indices[cluster_nr].indices.size());

    for (size_t index = 0; index < cluster_indices[cluster_nr].indices.size(); ++index) {
      cluster_cloud->points[index] = point_cloud->points[cluster_indices[cluster_nr].indices[index]];
    }

    cluster_clouds.push_back(cluster_cloud);
  } 
}

point_cloud_functions::ObjectROI PointCloudFunctions::getROI(PointCloudPtr original_point_cloud, PointCloudPtr cluster_cloud) {
  pcl::search::KdTree<Point>::Ptr search_tree(new pcl::search::KdTree<Point>());
  search_tree->setInputCloud(original_point_cloud);

  size_t width = original_point_cloud->width;
  size_t height = original_point_cloud->height;
  
  point_cloud_functions::ObjectROI roi;
  roi.left = width;
  roi.right = 0;
  roi.top = height; 
  roi.bottom = 0;

  float center_x = 0.0f;
  float center_y = 0.0f;
  float center_z = 0.0f;
  sensor_msgs::PointCloud2 publish_cloud;

  getSensorMessageFromPointCloud(cluster_cloud, publish_cloud);

  //point_cloud_publisher_.publish(publish_cloud); 

  for (auto point: cluster_cloud->points) {
    std::vector<int> indices;
    std::vector<float> sqrt_distances; // unused but required for function call

    search_tree->nearestKSearch(point, 1, indices, sqrt_distances);
    size_t index = indices[0];
    size_t x = index % width;
    size_t y = std::floor(index / width);

    if (x < roi.left) {
      roi.left = x;
    }

    if (x > roi.right) {
      roi.right = x;
    }

    if (y < roi.top) {
      roi.top = y;
    }

    if (y > roi.bottom) {
      roi.bottom = y;
    }

    center_x += point.x;
    center_y += point.y;
    center_z += point.z;
  }

  roi.center_x = center_x / cluster_cloud->points.size();
  roi.center_y = center_y / cluster_cloud->points.size();
  roi.center_z = center_z / cluster_cloud->points.size();
  return roi;
}