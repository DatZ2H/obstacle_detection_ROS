#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_ros/transforms.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>


class ObstacleDetection
{
public:
  ObstacleDetection()
  {
    // Khởi tạo các publisher và subscriber
    cloud_sub_ = nh_.subscribe("/camera/depth/color/points", 1, &ObstacleDetection::pointCloudCallback, this);
    filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("filtered_publisher", 1);
    downsampled_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("downsampled_publisher", 1);
    obstacles_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("obstacles_publisher", 1);
    clustered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("clustered_publisher", 1);
  }

  void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
  {
    // Chuyển đổi từ sensor_msgs::PointCloud2 sang pcl::PointCloud<pcl::PointXYZ>
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Bước 1: Lọc điểm dữ liệu trong một khoảng giá trị cho trước
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    passThroughFilter(cloud, filtered_cloud);
    filtered_pub_.publish(convertToROSMsg(filtered_cloud));

    // Bước 2: Giảm độ phân giải của điểm dữ liệu
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    voxelGridFilter(filtered_cloud, downsampled_cloud);
    downsampled_pub_.publish(convertToROSMsg(downsampled_cloud));

    // Bước 3: Loại bỏ sàn nhà
    pcl::PointCloud<pcl::PointXYZ>::Ptr floor_removed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    removeFloor(downsampled_cloud, floor_removed_cloud);
    obstacles_pub_.publish(convertToROSMsg(floor_removed_cloud));

    // Bước 4: Phân nhóm các điểm dữ liệu theo cụm
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
    euclideanClustering(floor_removed_cloud, clusters);

    // Xuất các cụm điểm dữ liệu
    pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& cluster : clusters)
    {
      *clustered_cloud += *cluster;
    }
    clustered_pub_.publish(convertToROSMsg(clustered_cloud));
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Publisher filtered_pub_;
  ros::Publisher downsampled_pub_;
  ros::Publisher obstacles_pub_;
  ros::Publisher clustered_pub_;

  void passThroughFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud)
  {
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(input_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 3.0);
    pass.filter(*filtered_cloud);
  }

  void voxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr& downsampled_cloud)
  {
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(input_cloud);
    voxel_grid.setLeafSize(0.01, 0.01, 0.01);
    voxel_grid.filter(*downsampled_cloud);
  }

void removeFloor(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr& floor_removed_cloud)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.01);

  int i = 0, nr_points = (int)input_cloud->points.size();
  while (input_cloud->points.size() > 0.3 * nr_points)
  {
    seg.setInputCloud(input_cloud);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0)
    {
      ROS_INFO("Could not estimate a planar model for the given dataset.");
      break;
    }

    extract.setInputCloud(input_cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_plane);

    extract.setNegative(true);
    extract.filter(*floor_removed_cloud);
    *input_cloud = *floor_removed_cloud;
  }
}


  void euclideanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
                           std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters)
  {
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(input_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.02);
    ec.setMinClusterSize(1000);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(input_cloud);
    ec.extract(cluster_indices);

    for (const auto& indices : cluster_indices)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
      for (const auto& index : indices.indices)
      {
        cluster->push_back((*input_cloud)[index]);
      }
      clusters.push_back(cluster);
    }
  }

  sensor_msgs::PointCloud2 convertToROSMsg(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
  {
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = "base_link";
    return cloud_msg;
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "obstacle_detection");
  
  

  ObstacleDetection obstacle_detection;


  ros::spin();

  return 0;
}
