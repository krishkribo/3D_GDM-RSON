#include "point_cloud_functions/point_cloud_functions.h"

bool PointCloudFunctions::getSensorMessage(sensor_msgs::PointCloud2 &point_cloud_message, std::string sensor_topic_name) {
  boost::shared_ptr<const sensor_msgs::PointCloud2> point_cloud_message_shared_ptr;

  point_cloud_message_shared_ptr = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(sensor_topic_name, ros::Duration(2.0));

  if (point_cloud_message_shared_ptr == nullptr) {
    ROS_WARN_STREAM("No point cloud message received on " << sensor_topic_name << " after 2 second");
    return false;
  }

  point_cloud_message = *point_cloud_message_shared_ptr;
  return true;
}

bool PointCloudFunctions::transformPointCloudTo(sensor_msgs::PointCloud2 &point_cloud_msg, 
                                                sensor_msgs::PointCloud2 &transformed_point_cloud_msg,
                                                std::string transform_to_link) {
  if (transform_to_link.size() == 0) {
    return false;
  }

  try {
    transform_listener_.waitForTransform(transform_to_link,
                                         point_cloud_msg.header.frame_id,
                                         ros::Time::now(),
                                         ros::Duration(1.0));
  } catch (...) {
    ROS_WARN_STREAM("PointCloudFunctions::transformPointCloudTo: transform from " << transform_to_link << " to " << point_cloud_msg.header.frame_id << " not available");
    return false;
  }

  if (!transform_listener_.canTransform(transform_to_link,
                                        point_cloud_msg.header.frame_id,
                                        point_cloud_msg.header.stamp)) {
    ROS_WARN_STREAM("PointCloudFunctions::transformPointCloudTo: can not transform " << transform_to_link << " to " << point_cloud_msg.header.frame_id);
    return false;
  }

  //ROS_INFO_STREAM("Transforming point cloud to " << transform_to_link);
  transformPointCloud(transform_to_link, point_cloud_msg, transformed_point_cloud_msg, transform_listener_);
  return true;
}

void PointCloudFunctions::getPointCloudFromSensorMessage(sensor_msgs::PointCloud2 &point_cloud_msg, PointCloudPtr point_cloud) {
  pcl::fromROSMsg(point_cloud_msg, *point_cloud);
}

void PointCloudFunctions::getSensorMessageFromPointCloud(PointCloudPtr point_cloud, sensor_msgs::PointCloud2 &point_cloud_msg) {
  pcl::toROSMsg(*point_cloud, point_cloud_msg);
}

bool PointCloudFunctions::getPointCloud(PointCloudPtr point_cloud, point_cloud_functions::GetObjectROIRequest &req) {
  PointCloudPtr new_point_cloud(new PointCloud());
  bool result = getPointCloud(point_cloud, new_point_cloud, req);

  if (result) { 
    point_cloud = new_point_cloud;
  } 

  return result;
}

bool PointCloudFunctions::getPointCloud(PointCloudPtr point_cloud, point_cloud_functions::FindTableRequest &req) {
  FilterOptions filter_options;
  filter_options.transform_to_link = req.transform_to_link;
  filter_options.filter_z = true;
  filter_options.min_z = req.min_z;
  filter_options.max_z = req.max_z;

  return getPointCloud(point_cloud, filter_options);
}

bool PointCloudFunctions::getPointCloud(PointCloudPtr point_cloud, point_cloud_functions::GetClosestPointRequest &req) {
  FilterOptions filter_options;
  filter_options.transform_to_link = req.transform_to_link;
  filter_options.filter_x = req.filter_x;
  filter_options.filter_y = req.filter_y;
  filter_options.filter_z = req.filter_z;
  filter_options.min_x = req.min_x;
  filter_options.max_x = req.max_x;
  filter_options.min_y = req.min_y;
  filter_options.max_z = req.max_y;
  filter_options.min_z = req.min_z;
  filter_options.max_z = req.max_z;

  return getPointCloud(point_cloud, filter_options);
}

bool PointCloudFunctions::getPointCloud(PointCloudPtr point_cloud, point_cloud_functions::GetCenterObjectsRequest &req) {
  FilterOptions filter_options;
  filter_options.transform_to_link = req.transform_to_link;
  filter_options.filter_x = req.filter_x;
  filter_options.filter_y = req.filter_y;
  filter_options.filter_z = req.filter_z;
  filter_options.min_x = req.min_x;
  filter_options.max_x = req.max_x;
  filter_options.min_y = req.min_y;
  filter_options.max_y = req.max_y;
  filter_options.min_z = req.min_z;
  filter_options.max_z = req.max_z;

  return getPointCloud(point_cloud, filter_options);
}

bool PointCloudFunctions::getPointCloud(PointCloudPtr point_cloud, point_cloud_functions::GetPlanarCoefficientsRequest &req) {
  FilterOptions filter_options; 
  filter_options.transform_to_link = req.transform_to;
  filter_options.filter_x = false;
  filter_options.filter_y = false;
  filter_options.filter_z = false;

  return getPointCloud(point_cloud, filter_options);
}

bool PointCloudFunctions::getPointCloud(PointCloudPtr point_cloud, point_cloud_functions::GetSurfaceRequest &req) {
  FilterOptions filter_options;
  filter_options.transform_to_link = req.transform_to;
  filter_options.filter_z = true;
  filter_options.min_z = req.min_z;
  filter_options.max_z = req.max_z;

  return getPointCloud(point_cloud, filter_options);
}

bool PointCloudFunctions::getPointCloud(PointCloudPtr point_cloud, FilterOptions filter_options) {
  sensor_msgs::PointCloud2 point_cloud_message;

  if (!getSensorMessage(point_cloud_message, sensor_topic_name_)) {
    ROS_WARN_STREAM("Could not get point cloud message from " << sensor_topic_name_);
    return false;
  }

  transformPointCloudTo(point_cloud_message, point_cloud_message, filter_options.transform_to_link);
  getPointCloudFromSensorMessage(point_cloud_message, point_cloud);

  if (filter_options.filter_x) {
    passThroughFilter(point_cloud, point_cloud, "x", filter_options.min_x, filter_options.max_x);
  }

  if (filter_options.filter_y) {
    passThroughFilter(point_cloud, point_cloud, "y", filter_options.min_y, filter_options.max_y);
  }

  if (filter_options.filter_z) {
    passThroughFilter(point_cloud, point_cloud, "z", filter_options.min_z, filter_options.max_z);
  }

  if (point_cloud->points.size() == 0) {
    return false;
  }

  return true;
}

bool PointCloudFunctions::getPointCloud(PointCloudPtr original_point_cloud, PointCloudPtr modified_point_cloud, point_cloud_functions::GetObjectROIRequest &req) {
  sensor_msgs::PointCloud2 point_cloud_message;
  std::string sensor_topic_name = req.sensor_topic_name == "" ? sensor_topic_name_ : req.sensor_topic_name;

  if (!getSensorMessage(point_cloud_message, sensor_topic_name)) {
    ROS_WARN_STREAM("Can not get point cloud message from " << sensor_topic_name);
    return false;
  }
  

  if (req.transform_to_link.size() > 0) {
    
    if (!transformPointCloudTo(point_cloud_message, point_cloud_message, req.transform_to_link)) {
      return false;
    }
  }

  getPointCloudFromSensorMessage(point_cloud_message, original_point_cloud);
  getPointCloudFromSensorMessage(point_cloud_message, modified_point_cloud); 
  
  if (req.filter_x) {
    passThroughFilter(modified_point_cloud, modified_point_cloud, "x", req.min_x, req.max_x);
  
    if (modified_point_cloud->points.size() == 0) {
      ROS_WARN_STREAM("No more points left after passThroughFilter X");
      return false; 
    }
  }

  if (req.filter_y) {
    passThroughFilter(modified_point_cloud, modified_point_cloud, "y", req.min_y, req.max_y);

    if (modified_point_cloud->points.size() == 0) {
      ROS_WARN_STREAM("No more points left after passThroughFilter Y");
      return false;
    }
  }

  if (req.filter_z) {
    passThroughFilter(modified_point_cloud, modified_point_cloud, "z", req.min_z, req.max_z);

    if (modified_point_cloud->points.size() == 0) {
      ROS_WARN_STREAM("No more points left after passThroughFilter Z");
      return false;
    }
  }
  
  return true;
}

Point PointCloudFunctions::findClosestPoint(PointCloudPtr point_cloud, Point search_point, int K) {
  pcl::search::KdTree<Point>::Ptr search_tree(new pcl::search::KdTree<Point>());
  search_tree->setInputCloud(point_cloud);
  std::vector<int> index_from_search;
  std::vector<float> squared_distance;

  search_tree->nearestKSearch(search_point, K, index_from_search, squared_distance);

  Point closest_point;
  closest_point.x = 0.0f;
  closest_point.y = 0.0f;
  closest_point.z = 0.0f;

  for (auto index: index_from_search) {
    Point temp_point = point_cloud->points[index];
    closest_point.x += temp_point.x;
    closest_point.y += temp_point.y;
    closest_point.z += temp_point.z;
  }

  closest_point.x /= index_from_search.size();
  closest_point.y /= index_from_search.size();
  closest_point.z /= index_from_search.size();

  return closest_point;
}

Point PointCloudFunctions::findClosestPoint(PointCloudPtr point_cloud) {
  pcl::search::KdTree<Point>::Ptr search_tree(new pcl::search::KdTree<Point>());
  search_tree->setInputCloud(point_cloud);

  size_t K = 1;
  std::vector<int> index_from_search;
  std::vector<float> squared_distance;
  Point search_point(0, 0, 0);
  search_tree->nearestKSearch(search_point, K, index_from_search, squared_distance);

  return point_cloud->points.at(index_from_search[0]);
}