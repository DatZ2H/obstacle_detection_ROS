#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_ros/transforms.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/String.h>
#include <pcl/point_cloud.h>

#include <pcl/search/kdtree.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <string>

class ObstacleDetection
{
public:
  ObstacleDetection() : nh_("~")
  {
    // Đăng ký các publisher
    filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("filtered_publisher", 1);
    downsampled_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("downsampled_publisher", 1);
    obstacles_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("obstacles_publisher", 1);
    clustered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("clustered_publisher", 1);
    output_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("output_point_cloud_topic", 1);
    safety_warn_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("safety_warn_cloud", 1);
    safety_protect_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("safety_protect_cloud", 1);
    safety_status_pub_ = nh_.advertise<std_msgs::String>("safety_status", 1);

    // Đăng ký subscriber nhận dữ liệu từ topic /camera/depth/color/points
    cloud_sub_ = nh_.subscribe("/camera/depth/color/points", 1, &ObstacleDetection::cloudCallback, this);

    // Đọc các tham số từ file launch hoặc sử dụng giá trị mặc định
    nh_.param<double>("pass_through_z_min", pass_through_z_min_, 0.0);
    nh_.param<double>("pass_through_z_max", pass_through_z_max_, 2.0);
    nh_.param<double>("pass_through_y_min", pass_through_y_min_, -1.0);
    nh_.param<double>("pass_through_y_max", pass_through_y_max_, 1.0);
    nh_.param<double>("voxel_leaf_size", voxel_leaf_size_, 0.01);
    nh_.param<double>("ground_seg_distance_threshold", ground_seg_distance_threshold_, 0.01);
    nh_.param<double>("cluster_tolerance", cluster_tolerance_, 0.02);
    nh_.param<int>("min_cluster_size", min_cluster_size_, 100);
    nh_.param<int>("max_cluster_size", max_cluster_size_, 25000);
    nh_.param<double>("safety_warn_size", safety_warn_size_, 0.1);
    nh_.param<double>("safety_protect_size", safety_protect_size_, 0.2);
    nh_.param<double>("safety_warn_position_x", safety_warn_position_(0), 0.0);
    nh_.param<double>("safety_warn_position_y", safety_warn_position_(1), 0.0);
    nh_.param<double>("safety_warn_position_z", safety_warn_position_(2), 1.0);
    nh_.param<double>("safety_protect_position_x", safety_protect_position_(0), 0.0);
    nh_.param<double>("safety_protect_position_y", safety_protect_position_(1), 0.0);
    nh_.param<double>("safety_protect_position_z", safety_protect_position_(2), 1.0);
    nh_.param<double>("collision_threshold", collision_threshold_, 0.1);
    nh_.param<int>("min_cluster_warn_size", min_cluster_warn_size_, 10);
    nh_.param<int>("min_cluster_protect_size", min_cluster_protect_size_, 10);
    nh_.param<int>("consecutive_warn_count", min_consecutive_warn_count_, 5);
    nh_.param<int>("consecutive_protect_count", min_consecutive_protect_count_, 5);

    displayParameters();
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
  {
    // Chuyển đổi sensor_msgs/PointCloud2 thành pcl::PointCloud<pcl::PointXYZ>
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Lọc dữ liệu theo trục z và y bằng PassThrough filter
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    passThroughFilter(cloud, filtered_cloud);

    // Giảm độ phân giải của điểm dữ liệu bằng VoxelGrid filter
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    voxelGridFilter(filtered_cloud, downsampled_cloud);

    // Áp dụng Ground segmentation algorithm để loại bỏ sàn nhà
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_removed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    segmentGround(downsampled_cloud, ground_removed_cloud);

    // Phân nhóm các điểm dữ liệu thành các cụm sử dụng Euclidean clustering
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
    euclideanClustering(ground_removed_cloud, clusters);

    // Xuất kết quả các bước xử lý qua các topic tương ứng
    filtered_pub_.publish(convertToPointCloud2(filtered_cloud));
    downsampled_pub_.publish(convertToPointCloud2(downsampled_cloud));
    obstacles_pub_.publish(convertToPointCloud2(ground_removed_cloud));
    clustered_pub_.publish(convertToPointCloud2(clusters[0]));

    // Kiểm tra vùng an toàn
    // Tạo cloud chứa các cụm vật thể
    pcl::PointCloud<pcl::PointXYZ>::Ptr safety_warn_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr safety_protect_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    computeSafetyZone(ground_removed_cloud, safety_warn_cloud, safety_protect_cloud);

    // Gửi thông tin vùng an toàn và trạng thái an toàn
    safety_warn_pub_.publish(convertToPointCloud2(safety_warn_cloud));
    safety_protect_pub_.publish(convertToPointCloud2(safety_protect_cloud));

    // Tính toán trạng thái an toàn
    std_msgs::String safety_status_msg;
    safety_status_msg.data = computeSafetyStatus(ground_removed_cloud, safety_warn_cloud, safety_protect_cloud);

    // Publish trạng thái an toàn
    safety_status_pub_.publish(safety_status_msg);

    // Gửi kết quả xử lý của PCL qua topic output_point_cloud_topic
    // output_pub_.publish(convertToPointCloud2(clusters[0]));
  }

  void passThroughFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out)
  {
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud_in);

    // Filter along the z-axis
    pass.setFilterFieldName("z");
    pass.setFilterLimits(pass_through_z_min_, pass_through_z_max_);
    pass.filter(*cloud_out);

    // Filter along the y-axis
    pass.setFilterFieldName("y");
    pass.setFilterLimits(pass_through_y_min_, pass_through_y_max_);
    pass.filter(*cloud_out);
  }

  void voxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out)
  {
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud_in);
    voxel_grid.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
    voxel_grid.filter(*cloud_out);
  }

  void segmentGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out)
  {
    // Set the segmentation parameters
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(ground_seg_distance_threshold_);
     seg.setMaxIterations(100);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // Set the segmentation parameters
    seg.setInputCloud(cloud_in);
    seg.segment(*inliers, *coefficients);
    // Extract the ground points
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_in);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud_out);
  }

  void euclideanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in,
                           std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clusters)
  {
    // Phân nhóm các điểm dữ liệu thành các cụm
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_in);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_in);

    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);
    // Tạo cloud chứa các cụm vật thể
    int cluster_label = 0;
    for (const auto &indices : cluster_indices)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
      for (const auto &index : indices.indices)
        cluster->points.push_back(cloud_in->points[index]);
      cluster->width = cluster->points.size();
      cluster->height = 1;
      cluster->is_dense = true;
      clusters.push_back(cluster);
    }
    cluster_label++;
  }

  void computeSafetyZone(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr &safety_warn_cloud,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr &safety_protect_cloud)
  {
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud_in);

    // Vùng cảnh báo
    pass.setFilterFieldName("x");
    pass.setFilterLimits(safety_warn_position_(0) - safety_warn_size_ / 2,
                         safety_warn_position_(0) + safety_warn_size_ / 2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_warn_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pass.filter(*filtered_warn_cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(safety_warn_position_(1) - safety_warn_size_ / 2,
                         safety_warn_position_(1) + safety_warn_size_ / 2);
    pass.filter(*safety_warn_cloud);

    // Vùng bảo vệ
    pass.setFilterFieldName("x");
    pass.setFilterLimits(safety_protect_position_(0) - safety_protect_size_ / 2,
                         safety_protect_position_(0) + safety_protect_size_ / 2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_protect_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pass.filter(*filtered_protect_cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(safety_protect_position_(1) - safety_protect_size_ / 2,
                         safety_protect_position_(1) + safety_protect_size_ / 2);
    pass.filter(*safety_protect_cloud);

    // Lọc dữ liệu theo trục z trong vùng an toàn
    pass.setFilterFieldName("z");
    pass.setFilterLimits(safety_warn_position_(2) - safety_warn_size_ / 2,
                         safety_warn_position_(2) + safety_warn_size_ / 2);
    pass.setInputCloud(filtered_warn_cloud);
    pass.filter(*safety_warn_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(safety_protect_position_(2) - safety_protect_size_ / 2,
                         safety_protect_position_(2) + safety_protect_size_ / 2);
    pass.setInputCloud(filtered_protect_cloud);
    pass.filter(*safety_protect_cloud);
  }

  std::string computeSafetyStatus(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr &safety_warn_cloud,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr &safety_protect_cloud)
  {

    int num_warn = safety_warn_cloud->size();
    int num_protect = safety_protect_cloud->size();

    // static int consecutive_warn_count = 0;
    // static int consecutive_protect_count = 0;

    std::string status = "Safe";
      if (num_warn >= min_cluster_warn_size_)
    {

      consecutive_warn_count++;
      // consecutive_protect_count = 0;

      if (consecutive_warn_count >= min_consecutive_warn_count_)
      {
        // ROS_WARN_STREAM("Safety zone breached! Proceed with caution.");
        status = "Warning";
      }
    }
    else
    {
      consecutive_warn_count = 0;
    }

    if (num_protect >= min_cluster_protect_size_)
    {

      consecutive_protect_count++;
      // consecutive_warn_count = 0;

      if (consecutive_protect_count >= min_consecutive_protect_count_)
      {
        //  ROS_WARN_STREAM("Safety zone breached! Take immediate action!");
        status = "Protected";
      }
    }
    else
    {
      consecutive_protect_count = 0;
    }
  

      
    


    return status;
  }

  sensor_msgs::PointCloud2 convertToPointCloud2(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
  {
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = "base_link";
    return cloud_msg;
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Publisher filtered_pub_;
  ros::Publisher downsampled_pub_;
  ros::Publisher obstacles_pub_;
  ros::Publisher clustered_pub_;
  ros::Publisher output_pub_;
  ros::Publisher safety_warn_pub_;
  ros::Publisher safety_protect_pub_;
  ros::Publisher safety_status_pub_;

  double pass_through_z_min_;
  double pass_through_z_max_;
  double pass_through_y_min_;
  double pass_through_y_max_;
  double voxel_leaf_size_;

  double ground_seg_distance_threshold_;
  double cluster_tolerance_;
  int min_cluster_size_;
  int max_cluster_size_;
  double collision_threshold_;
  int collision_detection_limit_;
  Eigen::Vector3d safety_warn_position_;
  double safety_warn_size_;
  Eigen::Vector3d safety_protect_position_;
  double safety_protect_size_;
  std::vector<int> collision_counter_;
  int min_cluster_warn_size_;
  int min_cluster_protect_size_;
  int min_consecutive_warn_count_;
  int min_consecutive_protect_count_;
  int consecutive_warn_count = 0;
  int consecutive_protect_count = 0;

  void displayParameters()
  {
    ROS_INFO("Obstacle Detection Parameters:");
    ROS_INFO("pass_through_z_min: %f", pass_through_z_min_);
    ROS_INFO("pass_through_z_max: %f", pass_through_z_max_);
    ROS_INFO("voxel_leaf_size: %f", voxel_leaf_size_);
    ROS_INFO("ground_seg_distance_threshold: %f", ground_seg_distance_threshold_);
    ROS_INFO("cluster_tolerance: %f", cluster_tolerance_);
    ROS_INFO("min_cluster_size: %d", min_cluster_size_);
    ROS_INFO("max_cluster_size: %d", max_cluster_size_);
    ROS_INFO("safety_warn_size: %f", safety_warn_size_);
    ROS_INFO("safety_protect_size: %f", safety_protect_size_);
    ROS_INFO("safety_warn_position: [%f, %f, %f]", safety_warn_position_(0), safety_warn_position_(1), safety_warn_position_(2));
    ROS_INFO("safety_protect_position: [%f, %f, %f]", safety_protect_position_(0), safety_protect_position_(1), safety_protect_position_(2));
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "obstacle_detection");
  ObstacleDetection obstacle_detection;
  ros::spin();
  return 0;
}
