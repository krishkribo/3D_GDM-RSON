#include "point_cloud_functions/point_cloud_functions.h"

bool PointCloudFunctions::getObjectROICallback(point_cloud_functions::GetObjectROIRequest &req, point_cloud_functions::GetObjectROIResponse &res) {
  PointCloudPtr original_point_cloud(new PointCloud());
  PointCloudPtr modified_point_cloud(new PointCloud());

  if (!getPointCloud(original_point_cloud, modified_point_cloud, req)) {
    return false;
  } 

  if (req.voxelize_cloud) {
    voxelizePointCloud(modified_point_cloud, modified_point_cloud, req.leaf_size);

    if (modified_point_cloud->points.size() == 0) {
      ROS_WARN_STREAM("No points in cloud after voxelization with leaf size: " << req.leaf_size);
      return false;
    }
  }
  
  segmentPointCloud(modified_point_cloud, modified_point_cloud, true, req.surface_threshold);

  sensor_msgs::PointCloud2 point_cloud_msg;
  getSensorMessageFromPointCloud(modified_point_cloud, point_cloud_msg);
  point_cloud_msg.header.frame_id = "new_frame_id";
  point_cloud_publisher_.publish(point_cloud_msg);

  std::vector<PointCloudPtr> cluster_point_clouds;
  getClusters(modified_point_cloud, cluster_point_clouds, req.min_cluster_points, req.max_cluster_points, req.cluster_distance);

  if (cluster_point_clouds.size() == 0) {
    ROS_WARN_STREAM("No clusters founds");
    return false;
  }
  
  for (auto cluster_point_cloud: cluster_point_clouds) {
    point_cloud_functions::ObjectROI roi;
    roi = getROI(original_point_cloud, cluster_point_cloud);
    res.object_rois.push_back(roi);
  } 
  
  return true;
}

bool PointCloudFunctions::voxelizePointCloudCallback(point_cloud_functions::VoxelizePointCloudRequest &req, point_cloud_functions::VoxelizePointCloudResponse &res) {
  PointCloudPtr point_cloud(new PointCloud());
  getPointCloudFromSensorMessage(req.point_cloud, point_cloud);
  voxelizePointCloud(point_cloud, point_cloud, req.leaf_size);

  if (point_cloud->points.size() == 0) {
    ROS_WARN_STREAM("No points left afer voxelization with leaf size: " << std::to_string(req.leaf_size));
  }

  sensor_msgs::PointCloud2 voxelized_point_cloud_message;
  getSensorMessageFromPointCloud(point_cloud, voxelized_point_cloud_message);
  res.voxelized_point_cloud = voxelized_point_cloud_message;
  return true;
}

bool PointCloudFunctions::findTableCallback(point_cloud_functions::FindTableRequest &req, point_cloud_functions::FindTableResponse &res) {
  PointCloudPtr point_cloud(new PointCloud());
 
  if (!getPointCloud(point_cloud, req)) {
    return false;
  }

  std::vector<PointCloudPtr> planes;
  float surface_threshold = 0.015f;

  //while (point_cloud->points.size() > 100) {
  PointCloudPtr plane_point_cloud(new PointCloud());
  segmentPointCloud(point_cloud, plane_point_cloud, false, surface_threshold);
  planes.push_back(plane_point_cloud);
  segmentPointCloud(point_cloud, point_cloud, true, surface_threshold);
  //}

  if (planes.size() == 0) {
    return false;
  }

  float closest_distance = std::numeric_limits<float>::max();
  float largest_size = 0;
  PointCloudPtr closest_plane(new PointCloud());
  PointCloudPtr largest_plane(new PointCloud());

  for (auto plane: planes) {
    //Point point = findClosestPoint(plane);
    float size = plane->points.size(); 
    //float distance = std::sqrt(std::pow(point.x, 2) + std::pow(point.y, 2));
    /**
    if (distance < closest_distance) {
      closest_distance = distance;
      closest_plane = plane;
    }
    **/

    if (size > largest_size) {
      largest_size = size;
      largest_plane = plane; 
    }
  }

  Point min_point, max_point;
  //pcl::getMinMax3D(*closest_plane, min_point, max_point);
  pcl::getMinMax3D(*largest_plane, min_point, max_point);
  res.min_point.x = min_point.x;
  res.min_point.y = min_point.y;
  res.min_point.z = min_point.z;
  res.max_point.x = max_point.x;
  res.max_point.y = max_point.y;
  res.max_point.z = max_point.z;
  
  return true;
}

bool PointCloudFunctions::getClosestPointCallback(point_cloud_functions::GetClosestPointRequest &req, point_cloud_functions::GetClosestPointResponse &res) {
  PointCloudPtr point_cloud(new PointCloud());

  if (!getPointCloud(point_cloud, req)) {
    return false;
  }  

  Point search_point;
  search_point.x = req.search_point.x;
  search_point.y = req.search_point.y;
  search_point.z = req.search_point.z; 
  Point closest_point = findClosestPoint(point_cloud, search_point);
  res.closest_point.x = closest_point.x;
  res.closest_point.y = closest_point.y;
  res.closest_point.z = closest_point.z;
  
  return true;
}

bool PointCloudFunctions::getCenterObjectsCallback(point_cloud_functions::GetCenterObjectsRequest &req, point_cloud_functions::GetCenterObjectsResponse &res) {
  PointCloudPtr point_cloud(new PointCloud());  

  if (!getPointCloud(point_cloud, req)) {
    return false;
  }
  
  //point_cloud_publisher_.publish(point_cloud); 
  float surface_threshold = 0.015f;
  // remove table surface
  PointCloudPtr table_cloud(new PointCloud());
  segmentPointCloud(point_cloud, table_cloud, false, surface_threshold);
  segmentPointCloud(point_cloud, point_cloud, true, surface_threshold);

  if (point_cloud->points.size() == 0) {
    return false;
  } 

  Point min_point_table, max_point_table;
  pcl::getMinMax3D(*table_cloud, min_point_table, max_point_table);

  std::vector<PointCloudPtr> clusters;
  getClusters(point_cloud, clusters, 600, 50000, 0.05f);

  if (clusters.size() == 0) {
    return false;
  }

  Point center_point;
  center_point.x = 0.0f;
  center_point.y = 0.0f;
  center_point.z = 0.0f;
  size_t total_points = 0;

  PointCloudPtr temp_cloud(new PointCloud());

  for (auto cluster: clusters) {
    ROS_INFO_STREAM("Cluster points: " << cluster->points.size());
    Point min_point, max_point;
    pcl::getMinMax3D(*cluster, min_point, max_point);

    if (min_point.z >= min_point_table.z) {
      for (auto point: cluster->points) {
        temp_cloud->points.push_back(point);
        center_point.x += point.x;
        center_point.y += point.y;
        center_point.z += point.z;
        ++total_points;
      }
    }
  }
	
	temp_cloud->header.frame_id = "new_frame_id";
  //point_cloud_publisher_.publish(*temp_cloud);

  center_point.x /= total_points;
  center_point.y /= total_points;
  center_point.z /= total_points;

  res.center_point.x = center_point.x;
  res.center_point.y = center_point.y;
  res.center_point.z = center_point.z;
  return true;
}

bool PointCloudFunctions::getPlanarCoefficientsCallback(point_cloud_functions::GetPlanarCoefficientsRequest &req, point_cloud_functions::GetPlanarCoefficientsResponse &res) {
  PointCloudPtr point_cloud(new PointCloud);
  
  if (!getPointCloud(point_cloud, req)) {
    ROS_INFO_STREAM("Failed getting point cloud");
    return false;
  }

  pcl::ModelCoefficients::Ptr model_coefficients(new pcl::ModelCoefficients());
  getPlanarCoefficients(point_cloud, req.surface_threshold, model_coefficients);
  res.a = model_coefficients->values.at(0);
  res.b = model_coefficients->values.at(1);
  res.c = model_coefficients->values.at(2);
  res.d = model_coefficients->values.at(3);
  return true;
}

bool PointCloudFunctions::getSurfaceCallback(point_cloud_functions::GetSurfaceRequest &req, point_cloud_functions::GetSurfaceResponse &res) {
  PointCloudPtr point_cloud(new PointCloud);
  PointCloudPtr segmented_point_cloud(new PointCloud);

  if (!getPointCloud(point_cloud, req)) {
    ROS_INFO_STREAM("Failed getting point cloud");
    return false;
  }

  segmentPointCloud(point_cloud, segmented_point_cloud, false, req.surface_threshold);

  Point min_point, max_point;
  pcl::getMinMax3D(*segmented_point_cloud, min_point, max_point);
  res.min_point.x = min_point.x;
  res.min_point.y = min_point.y;
  res.min_point.z = min_point.z;
  res.max_point.x = max_point.x;
  res.max_point.y = max_point.y;
  res.max_point.z = max_point.z;

  return true;
  
}
