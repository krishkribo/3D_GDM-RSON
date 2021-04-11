#! /usr/bin/python3

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import rospy
import pcl


def callback_pointcloud(ros_cloud):
    """assert isinstance(data,PointCloud2)
    get_cloud_data = point_cloud2.read_points(data,skip_nans=True)
    rospy.sleep(1)
    print(get_cloud_data)
    print([ i for i in get_cloud_data])"""
    points_list = []

    for data in point_cloud2.read_points(ros_cloud, skip_nans=True):
        points_list.append([data[0], data[1], data[2], data[3]])

    pcl_data = pcl.PointCloud_PointXYZRGB()
    pcl_data.from_list(points_list)

    print(pcl_data)
    rospy.sleep(1.0)

if __name__ == "__main__":
    rospy.init_node("test_point_cloud_node")
    rospy.Subscriber("/camera/depth/points",PointCloud2,callback_pointcloud)
    rospy.spin()
