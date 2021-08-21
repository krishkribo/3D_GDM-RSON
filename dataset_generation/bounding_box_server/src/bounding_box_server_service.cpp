#include "bounding_box_server/bounding_box_server.h"

bool BoundingBoxServer::getBoundingBoxesCallback(bounding_box_server::GetBoundingBoxRequest &req, bounding_box_server::GetBoundingBoxResponse &res) {
  sensor_msgs::PointCloud2 point_cloud_message;

  if (!getPointCloudMessage(point_cloud_message)) {
    return false;
  }

  if (!transformPointCloudMessageToLink(point_cloud_message, point_cloud_message, req.transform_to)) {
    return false;
  }

  PointCloudPtr point_cloud(new PointCloud());
  pcl::fromROSMsg(point_cloud_message, *point_cloud);

  voxelizePointCloud(point_cloud, point_cloud);
  passThroughFilter(point_cloud, point_cloud, "x", req.min_x, req.max_x);
  removeFloor(point_cloud, point_cloud, req.surface_threshold); // Remove the planar (should be floor) surface

  std::vector<PointCloudPtr> cluster_clouds;
  extractClusters(point_cloud, cluster_clouds, req.cluster_threshold); // Extract the clusters and keep them in a dynamic array (vector)

  if (cluster_clouds.size() == 0) { // Could be that no clusters are found
    ROS_WARN("No clusters found");
    return false;
  }

  std::vector<bounding_box_server::BoundingBox> bounding_boxes;

  for (auto cluster_cloud: cluster_clouds) { // For each Point Cloud in the dynamic array (vector) do
    PointCloudPtr projected_point_cloud(new PointCloud());
    projectPointCloudOnPlane(cluster_cloud, projected_point_cloud); // Flatten the Point Cloud on the planar surface

    Eigen::Matrix3f eigen_vectors = getEigenVectors(projected_point_cloud); // Retreive the 3 eigen vectors, in a column based matrix
    float angle = getAngle(eigen_vectors.col(0));
    
    Eigen::Vector3f centroid_vector = getCenterPointCloud(cluster_cloud); // Calculate the center of the Point Cloud, in a Vector3f format
    
    PointCloudPtr centered_point_cloud(new PointCloud());
    transformPointCloudToCenter(cluster_cloud, centered_point_cloud, centroid_vector, angle); // Rotate and Translate the Point Cloud to the origin (0, 0, 0)

    bounding_box_server::BoundingBox bounding_box;
    bounding_box.x = centroid_vector.x();
    bounding_box.y = centroid_vector.y();
    bounding_box.z = centroid_vector.z();
    bounding_box.yaw = angle;

    getDimensions(centered_point_cloud, bounding_box); // Calculate the length, width, and height of the Transformed Point Cloud
    bounding_boxes.push_back(bounding_box);
  }

  res.bounding_boxes = bounding_boxes;
  bounding_box_publisher_.publishBoundingBoxes(bounding_boxes, req.transform_to);
  return true;
}
