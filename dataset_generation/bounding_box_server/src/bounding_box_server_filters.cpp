#include "bounding_box_server/bounding_box_server.h"

//! Filters parts of the Point Cloud per axis based on min and max values.
/**
 * Uses the pcl::PassThrough filter to filter out parts of the Point Cloud. 
 * All points on the specified axis between min_value and max_value will be contained in the output Point Cloud, 
 * point below min_value and above max_value will therefore be removed.
 * \param point_cloud Pointer to input Point Cloud.
 * \param filtered_point_cloud Pointer to the output filtered Point Cloud.
 * \param field_name the axis on which to filter, either "x", "y", or "z".
 * \param min_value the min value off all the filtered points.
 * \param max_value the max value off all the filtered points.
 */
void BoundingBoxServer::passThroughFilter(PointCloudPtr point_cloud, PointCloudPtr filtered_point_cloud,
                                          std::string field_name, float min_value, float max_value) {
  pass_through_filter_.setInputCloud(point_cloud); // Pointer to the input Point Cloud
  pass_through_filter_.setFilterFieldName(field_name); // Indicates on which axis to filter
  pass_through_filter_.setFilterLimits(min_value, max_value); 
  pass_through_filter_.filter(*filtered_point_cloud);  // The output of the filter                               
}

//! Removes a flat surface from the Point Cloud.
/**
 * Removes 1 flat surface, which should be the floor, from the Point Cloud. It uses RANSAC to find points belonging to a possible 
 * planar surface, and checks if the points lays within a theshold to determine if it belongs to the surface or not. 
 * \param point_cloud Pointer to the input Point Cloud, should only contain one large planar surface (the floor), assumes there is no floor visible.
 * \param floorless_point_cloud Pointer to the Point Cloud without a floor surface in it.
 * \param distance_threshold determines the threshold in meters when a points belongs to a possible planar surface or not.
 */
void BoundingBoxServer::removeFloor(PointCloudPtr point_cloud, PointCloudPtr floorless_point_cloud, float distance_threshold) {
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients); 
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices); 


  /*
  sac_segmentation_ is a pcl::SACSegmentation<Point> class object, 
  https://pointclouds.org/documentation/classpcl_1_1_s_a_c_segmentation.html

  Which inherits the PCLBase class
  https://pointclouds.org/documentation/classpcl_1_1_p_c_l_base.html

  - You need to set the input point cloud, point_cloud.
  - You need to set a distance threshold, distance_threshold, how close is a point when it still belongs to the surface
  - Set the method type to pcl::SAC_RANSAC
  - Set the model type to pcl::SACMODEL_PLANE
  - Segment using the inliers and coefficients objects (they are pointers so use a * before the name)
    Whilst both parameters are required, only inliers will be used later
    The inliers object should now contain all the indexes of the points that are part of the floor 
  */

 sac_segmentation_.setInputCloud(point_cloud);
 sac_segmentation_.setDistanceThreshold(distance_threshold);
 sac_segmentation_.setMethodType(pcl::SAC_RANSAC);
 sac_segmentation_.setModelType(pcl::SACMODEL_PLANE);
 sac_segmentation_.segment(*inliers,*coefficients);

  /* 
  extract_indices_ is a pcl::ExtractIndices<Point> class object,
  https://pointclouds.org/documentation/classpcl_1_1_extract_indices.html

  which inherits the PCLBase class,

  and the pcl::FilterIndices<Point> class,
  https://pointclouds.org/documentation/classpcl_1_1_filter_indices.html

  - You need to set the input point cloud
  - You need to set the indicies 
  - You need to set whether the to keep the indices, or remove the indices from the point cloud
  - Filter the point cloud, and write the new point cloud to *floorless_point_cloud

  */

 extract_indices_.setInputCloud(point_cloud);
 extract_indices_.setIndices(inliers);
 extract_indices_.setNegative(true);
 extract_indices_.filter(*floorless_point_cloud);
  
}

//! Given a Point Cloud, extract the clusters as seperate Point Clouds.
/**
 * Given a Point Cloud extract the clusters and put them in their own Point Cloud. 
 * \param point_cloud Pointer to the input Point Cloud. The Point Cloud should have the floor surface removed.
 * \param clusters a vector of Pointers to Point Clouds, each Point Cloud represents a cluster.
 * \param cluster_tolerance the distance between two points that indices when they become their seperate cluster.
 */
void BoundingBoxServer::extractClusters(PointCloudPtr point_cloud, std::vector<PointCloudPtr> &clusters, float cluster_tolerance) {
  pcl::search::KdTree<Point>::Ptr search_tree(new pcl::search::KdTree<Point>()); // Used for quickly searching the Point Cloud structure
  search_tree->setInputCloud(point_cloud); // Pointer to the input Point Cloud 

  std::vector<pcl::PointIndices> cluster_indices;
  
  /*
  cluster_extraction_ is a pcl::EuclideanClusterExtraction<Point> class object,
  https://pointclouds.org/documentation/classpcl_1_1_euclidean_cluster_extraction.html

  which inherits the PCLBase class,
  https://pointclouds.org/documentation/classpcl_1_1_p_c_l_base.html

  - Set the input point cloud, point_cloud
  - Set the search method, search_tree
  - Set the cluster tolerance, cluster_tolerance
  - Set the min and max cluster sizes, I recommend between 10 and 50000
  - Extract the clusters with the parameter, cluster_indices

    cluster_indices is now a vector, where each items is another vector of all the indices that belong to a single cluster
  */
  cluster_extraction_.setInputCloud(point_cloud);
  cluster_extraction_.setSearchMethod(search_tree);
  cluster_extraction_.setClusterTolerance(cluster_tolerance);
  cluster_extraction_.setMaxClusterSize(50000);
  cluster_extraction_.setMinClusterSize(10);
  cluster_extraction_.extract(cluster_indices);
  /*
  for each cluster in cluster_indices, do 
    - create a new point cloud ptr, PointCloudPtr cluster_cloud(new PointCloud())

    for each index value of the PointIndices, do 
      - add the point at position index, from the point_cloud, to the cluster_cloud.
      You can access the points via e.g. point_cloud->points, points is a vector, from which you can get an item via index, points[index], 
      or add a new item at the back, points.push_back(point)

    Add the newly create cluster_cloud to the clusters vector (parameter for the function), use push_back
  */ 
  std::vector<pcl::PointIndices>::const_iterator cluster;
  std::vector<int>::const_iterator index;
  for (cluster = cluster_indices.begin(); cluster != cluster_indices.end(); ++cluster)
  {
    PointCloudPtr cluster_cloud(new PointCloud());
    for (index = cluster->indices.begin(); index != cluster->indices.end(); index++)
    {
      cluster_cloud->points.push_back(point_cloud->points[*index]);
    }
    clusters.push_back(cluster_cloud);
  }
}

//! Voxelizes the Point Cloud.
/**
 * Reduces the Point Cloud size by creating voxels of multiple points based on the leaf size, each points within 
 * the voxel will be indicated just by that one voxel. This will mostly help to reduce the calculation time of other 
 * algorithms.
 * \param point_cloud Pointer to the Point Cloud.
 * \param voxelized_point_cloud Pointer to the voxelized Point Cloud based on leaf_size parameter.
 * \param leaf_size indices how large the voxels are.
 */
void BoundingBoxServer::voxelizePointCloud(PointCloudPtr point_cloud, PointCloudPtr voxelized_point_cloud, float leaf_size) {
  voxel_grid_.setInputCloud(point_cloud); // Pointer to the input Point Cloud
  voxel_grid_.setLeafSize(leaf_size, leaf_size, leaf_size); // The leaf size of the voxel, each point in this voxel will be indicated by 1 point
  voxel_grid_.filter(*voxelized_point_cloud); // The reduced Point Cloud 
}
