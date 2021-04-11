#include "bounding_box_server/bounding_box_publisher.h"

//! Constructor.
/** 
 * Creates two publishers, one to publish the visualizations markers so the bounding box can be visualized in Rviz, 
 * the other to make the bounding boxes available for use somewhere else.
 * Setups the offset array for creating the lines for the bounding box.
 * \param nh a NodeHandle created in main 
 */
BoundingBoxPublisher::BoundingBoxPublisher(ros::NodeHandle &nh) : 
    nh_(nh) {
  bounding_box_marker_array_publisher_ = nh_.advertise<visualization_msgs::MarkerArray>("/bounding_box_markers", 1);
  bounding_box_publisher_ = nh_.advertise<bounding_box_server::BoundingBoxes>("/bounding_boxes", 1);

  offsets_ = { // The off sets directions for drawing the lines
      {-1, -1, -1, 1, -1, -1},
      {-1, 1, -1, 1, 1, -1},
      {-1, -1, 1, 1, -1, 1},
      {-1, 1, 1, 1, 1, 1},
      {-1, 1, -1, -1, -1, -1},
      {1, 1, -1, 1, -1, -1},
      {-1, 1, 1, -1, -1, 1},
      {1, 1, 1, 1, -1, 1},
      {-1, 1, -1, -1, 1, 1},
      {1, 1, -1, 1, 1, 1},
      {-1, -1, -1, -1, -1, 1},
      {1, -1, -1, 1, -1, 1}
  };
}

//! Publish to bounding box message and the visualization markers 
/**
 * Creates a bounding box of lines and publishes this, also publishes the bounding box information.
 * \param bounding_boxes vector holding the bounding boxes information.
 */
void BoundingBoxPublisher::publishBoundingBoxes(std::vector<bounding_box_server::BoundingBox> bounding_boxes, std::string transform_to) {
  visualization_msgs::MarkerArray marker_array;

  size_t marker_id = 0; // Each marker needs to have a unique ID

  for (auto bounding_box: bounding_boxes) {
    visualization_msgs::Marker marker;
		marker.header.frame_id = transform_to; // Transform from the "/root" frame of reference (used to transform the Point Cloud to)
		marker.header.stamp = ros::Time::now();
		marker.type = visualization_msgs::Marker::LINE_LIST;
		marker.lifetime = ros::Duration(3.0); // Keep the lines a life (visible) for 3 seconds 
		marker.id = marker_id;
		++marker_id; // Create new id value
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = bounding_box.x;
		marker.pose.position.y = bounding_box.y;
		marker.pose.position.z = bounding_box.z;
		tf::Quaternion quaternion;
		quaternion.setRPY(0, 0, bounding_box.yaw); // Create Quaternion from Roll Pitch Yaw value (in radians)
		marker.pose.orientation.x = quaternion[0];
		marker.pose.orientation.y = quaternion[1];
		marker.pose.orientation.z = quaternion[2];
		marker.pose.orientation.w = quaternion[3];
		
		marker.scale.x = 0.001; // size of line
		marker.color.r = 1.0; // Red color 
		marker.color.a = 1.0; // Alpha level, to make the line actually visible

    for (auto offset: offsets_) { // For each off set do
      geometry_msgs::Point point;
      point.x = offset[0] * bounding_box.length/2.0;
      point.y = offset[1] * bounding_box.width/2.0;
      point.z = offset[2] * bounding_box.height/2.0;
      marker.points.push_back(point);
      point.x = offset[3] * bounding_box.length/2.0;
      point.y = offset[4] * bounding_box.width/2.0;
      point.z = offset[5] * bounding_box.height/2.0;
      marker.points.push_back(point);
    }	

    marker_array.markers.push_back(marker);
  }

  bounding_box_marker_array_publisher_.publish(marker_array);
  bounding_box_server::BoundingBoxes bounding_boxes_msg;
  bounding_boxes_msg.bounding_boxes = bounding_boxes;
  bounding_box_publisher_.publish(bounding_boxes_msg);
}
