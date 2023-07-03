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
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
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
    pcl::PointCloud<pcl::PointXYZ>::Ptr safety_warn_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr safety_protect_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    computeSafetyZone(ground_removed_cloud, safety_warn_cloud, safety_protect_cloud);

    // Gửi thông tin vùng an toàn và trạng thái an toàn
    safety_warn_pub_.publish(convertToPointCloud2(safety_warn_cloud));
    safety_protect_pub_.publish(convertToPointCloud2(safety_protect_cloud));
    std_msgs::String safety_status_msg;
    safety_status_msg.data = computeSafetyStatus(safety_warn_cloud, safety_protect_cloud);
    safety_status_pub_.publish(safety_status_msg);

    // Gửi kết quả xử lý của PCL qua topic output_point_cloud_topic
   // output_pub_.publish(convertToPointCloud2(clusters[0]));
  }

  void passThroughFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out)
  {
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud_in);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(pass_through_z_min_, pass_through_z_max_);
    pass.filter(*cloud_out);
  }

  void voxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out)
  {
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud_in);
    voxel_grid.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
    voxel_grid.filter(*cloud_out);
  }

  void segmentGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out)
  {
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(ground_seg_distance_threshold_);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    seg.setInputCloud(cloud_in);
    seg.segment(*inliers, *coefficients);

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_in);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud_out);
  }

  void euclideanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                           std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters)
  {
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

    for (const auto& indices : cluster_indices)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
      for (const auto& index : indices.indices)
        cluster->points.push_back(cloud_in->points[index]);
      cluster->width = cluster->points.size();
      cluster->height = 1;
      cluster->is_dense = true;
      clusters.push_back(cluster);
    }
  }

  void computeSafetyZone(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr& safety_warn_cloud,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr& safety_protect_cloud)
  {
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud_in);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(safety_warn_position_(0) - safety_warn_size_ / 2,
                         safety_warn_position_(0) + safety_warn_size_ / 2);
    pass.filter(*safety_warn_cloud);

    pass.setFilterLimits(safety_protect_position_(0) - safety_protect_size_ / 2,
                         safety_protect_position_(0) + safety_protect_size_ / 2);
    pass.filter(*safety_protect_cloud);
  }

  std::string computeSafetyStatus(const pcl::PointCloud<pcl::PointXYZ>::Ptr& safety_warn_cloud,
                                  const pcl::PointCloud<pcl::PointXYZ>::Ptr& safety_protect_cloud)
  {
    if (safety_protect_cloud->empty())
      return "Safe";
    else if (safety_warn_cloud->empty())
      return "Protect";
    else
      return "Warn";
  }

  sensor_msgs::PointCloud2 convertToPointCloud2(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
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
  double voxel_leaf_size_;
  double ground_seg_distance_threshold_;
  double cluster_tolerance_;
  int min_cluster_size_;
  int max_cluster_size_;
  double safety_warn_size_;
  double safety_protect_size_;
  Eigen::Vector3d safety_warn_position_;
  Eigen::Vector3d safety_protect_position_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "obstacle_detection");
  ObstacleDetection obstacle_detection;
  ros::spin();
  return 0;
}
