#include <geometry_msgs/TransformStamped.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <string>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>

class ObstacleDetection
{
public:
  ObstacleDetection() : nh_("~")
  {
    // Đăng ký các publisher
    raw_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("raw_publisher", 1);
    filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("filtered_publisher", 1);
    downsampled_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("downsampled_publisher", 1);
    conditionalOR_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("radiusoutlierremoval_publisher", 1);
    radiusOR_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("radiusoutlierremoval_publisher", 1);
    denoised_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("denoised_publisher", 1);
    obstacles_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("obstacles_publisher", 1);
    clustered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("clustered_publisher", 1);
    output_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("output_point_cloud_topic", 1);
    safety_warn_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("safety_warn_cloud", 1);
    safety_protect_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("safety_protect_cloud", 1);
    safety_status_pub_ = nh_.advertise<std_msgs::String>("safety_status", 1);
    distance_pub_ = nh_.advertise<std_msgs::Float32>("distance_publisher", 1);

    // Đăng ký subscriber nhận dữ liệu từ topic /camera/depth/color/points
    cloud_sub_ = nh_.subscribe("/camera/depth/color/points", 1, &ObstacleDetection::cloudCallback, this);

    // Đọc các tham số từ file launch hoặc sử dụng giá trị mặc định
    nh_.param<double>("pass_through_x_min", pass_through_x_min_, -1.5);
    nh_.param<double>("pass_through_x_max", pass_through_x_max_, 1.5);
    nh_.param<double>("pass_through_y_min", pass_through_y_min_, -1.5);
    nh_.param<double>("pass_through_y_max", pass_through_y_max_, 0.25);
    nh_.param<double>("pass_through_z_min", pass_through_z_min_, 0.0);
    nh_.param<double>("pass_through_z_max", pass_through_z_max_, 3.0);
    
    nh_.param<double>("voxel_leaf_size", voxel_leaf_size_, 0.01);

    nh_.param("condition_min_z", condition_min_z_, 0.0);
    nh_.param("condition_max_z", condition_max_z_, 2.0);
    nh_.param("radius_search", radius_search_, 0.05);
    nh_.param("min_neighbors_in_radius", min_neighbors_in_radius_, 10);

    nh_.param<int>("ground_seg_max_iterations", ground_seg_max_iterations_, 50);
    nh_.param<double>("ground_seg_distance_threshold", ground_seg_distance_threshold_, 0.05);

    nh_.param<double>("cluster_tolerance", cluster_tolerance_, 0.02);
    nh_.param<int>("min_cluster_size", min_cluster_size_, 100);
    nh_.param<int>("max_cluster_size", max_cluster_size_, 25000);

    nh_.param<double>("safety_warn_size", safety_warn_size_, 1.2);
    nh_.param<double>("safety_protect_size", safety_protect_size_, 0.9);
    nh_.param<double>("safety_warn_position_x", safety_warn_position_(0), 0.0);
    nh_.param<double>("safety_warn_position_y", safety_warn_position_(1), 0.0);
    nh_.param<double>("safety_warn_position_z", safety_warn_position_(2), 0.0);
    nh_.param<double>("safety_protect_position_x", safety_protect_position_(0), 0.0);
    nh_.param<double>("safety_protect_position_y", safety_protect_position_(1), 0.0);
    nh_.param<double>("safety_protect_position_z", safety_protect_position_(2), 0.0);

    nh_.param<int>("min_cluster_warn_size", min_cluster_warn_size_, 10);
    nh_.param<int>("min_cluster_protect_size", min_cluster_protect_size_, 10);
    nh_.param<int>("min_consecutive_warn_count", min_consecutive_warn_count_, 5);
    nh_.param<int>("min_consecutive_protect_count", min_consecutive_protect_count_, 5);

    displayParameters();
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
  {
    // Chuyển đổi sensor_msgs/PointCloud2 thành pcl::PointCloud<pcl::PointXYZRGB>
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    raw_pub_.publish(convertToPointCloud2(cloud));

    // Lọc dữ liệu theo trục z và y bằng PassThrough filter
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    passThroughFilter(cloud, filtered_cloud);

    // Giảm độ phân giải của điểm dữ liệu bằng VoxelGrid filter
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    voxelGridFilter(filtered_cloud, downsampled_cloud);

    // Remove outliers using Conditional Outlier Removal
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_outliers_removed(new pcl::PointCloud<pcl::PointXYZRGB>);
    // conditionalOutlierRemoval(downsampled_cloud, cloud_outliers_removed);

    // Remove outliers using Radius Outlier Removal
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_outliers_removed2(new pcl::PointCloud<pcl::PointXYZRGB>);
    // radiusOutlierRemoval(downsampled_cloud, cloud_outliers_removed2);

    // Apply StatisticalOutlierRemoval filter to remove outliers
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr denoised_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // statisticalOutlierRemoval(downsampled_cloud, denoised_cloud);

    // Áp dụng Ground segmentation algorithm để loại bỏ sàn nhà
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr ground_removed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // segmentGround(downsampled_cloud, ground_removed_cloud);

    // Phân nhóm các điểm dữ liệu thành các cụm sử dụng Euclidean clustering
    //   std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters;
    //  euclideanClustering(ground_removed_cloud, clusters);

    // // Xuất kết quả các bước xử lý qua các topic tương ứng
    // filtered_pub_.publish(convertToPointCloud2(filtered_cloud));
    // downsampled_pub_.publish(convertToPointCloud2(downsampled_cloud));
    // denoised_pub_.publish(convertToPointCloud2(denoised_cloud));
    // obstacles_pub_.publish(convertToPointCloud2(ground_removed_cloud));
    // clustered_pub_.publish(convertToPointCloud2(clusters[0]));

    // Kiểm tra vùng an toàn
    // Tạo cloud chứa các cụm vật thể
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr safety_warn_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr safety_protect_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    computeSafetyZone(downsampled_cloud, safety_warn_cloud, safety_protect_cloud);

    // Gửi thông tin vùng an toàn và trạng thái an toàn
    safety_warn_pub_.publish(convertToPointCloud2(safety_warn_cloud));
    safety_protect_pub_.publish(convertToPointCloud2(safety_protect_cloud));

    // Tính toán trạng thái an toàn
    std_msgs::String safety_status_msg;
    safety_status_msg.data = computeSafetyStatus(downsampled_cloud, safety_warn_cloud, safety_protect_cloud);

    // Publish trạng thái an toàn
    safety_status_pub_.publish(safety_status_msg);

    computeDistanceZ(safety_warn_cloud,safety_protect_cloud);

    // Gửi kết quả xử lý của PCL qua topic output_point_cloud_topic
    // output_pub_.publish(convertToPointCloud2(clusters[0]));
  }

 void passThroughFilter(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_out)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  // Lọc dữ liệu theo trục x
  pcl::PassThrough<pcl::PointXYZRGB> pass_x;
  pass_x.setInputCloud(cloud_in);
  pass_x.setFilterFieldName("x");
  pass_x.setFilterLimits(pass_through_x_min_, pass_through_x_max_);
  pass_x.filter(*filtered_cloud);

  // Lọc dữ liệu theo trục y
  pcl::PassThrough<pcl::PointXYZRGB> pass_y;
  pass_y.setInputCloud(filtered_cloud);
  pass_y.setFilterFieldName("y");
  pass_y.setFilterLimits(pass_through_y_min_, pass_through_y_max_);
  pass_y.filter(*filtered_cloud);

  // Lọc dữ liệu theo trục z
  pcl::PassThrough<pcl::PointXYZRGB> pass_z;
  pass_z.setInputCloud(filtered_cloud);
  pass_z.setFilterFieldName("z");
  pass_z.setFilterLimits(pass_through_z_min_, pass_through_z_max_);
  pass_z.filter(*cloud_out);

  // Xuất kết quả qua topic filtered_pub_
  filtered_pub_.publish(convertToPointCloud2(cloud_out));
}


  void voxelGridFilter(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_out)
  {
    // Tạo một đối tượng VoxelGrid filter
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;

    // Thiết lập đầu vào cho bộ lọc VoxelGrid
    voxel_grid.setInputCloud(cloud_in);

    // Thiết lập kích thước leaf (kích thước của voxel)
    voxel_grid.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);

    // Áp dụng bộ lọc VoxelGrid
    voxel_grid.filter(*cloud_out);

    // Xuất kết quả qua topic downsampled_pub_
    downsampled_pub_.publish(convertToPointCloud2(cloud_out));
  }

  void statisticalOutlierRemoval(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_out)
  {
    // Tạo một đối tượng StatisticalOutlierRemoval filter
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;

    // Thiết lập đầu vào cho bộ lọc StatisticalOutlierRemoval
    sor.setInputCloud(cloud_in);

    // Thiết lập số lượng hàng xóm gần nhất để tính khoảng cách trung bình (mean distance)
    sor.setMeanK(50);

    // Thiết lập ngưỡng nhân chuẩn độ lệch tiêu chuẩn (standard deviation multiplier threshold)
    sor.setStddevMulThresh(1.0);

    // Áp dụng bộ lọc StatisticalOutlierRemoval
    sor.filter(*cloud_out);

    // Xuất kết quả qua topic denoised_pub_
    denoised_pub_.publish(convertToPointCloud2(cloud_out));
  }

  void conditionalOutlierRemoval(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_out)
  {
    // Tạo một đối tượng ConditionAnd
    pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr condition(new pcl::ConditionAnd<pcl::PointXYZRGB>());

    // Thêm điều kiện so sánh cho trường z
    condition->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
        new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::GT, condition_min_z_)));
    condition->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
        new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::LT, condition_max_z_)));

    // Tạo một đối tượng ConditionalRemoval filter
    pcl::ConditionalRemoval<pcl::PointXYZRGB> conditional_removal;

    // Thiết lập đầu vào cho bộ lọc ConditionalRemoval
    conditional_removal.setInputCloud(cloud_in);

    // Thiết lập điều kiện cho bộ lọc ConditionalRemoval
    conditional_removal.setCondition(condition);

    // Thiết lập cờ keepOrganized để giữ nguyên cấu trúc của điểm đám mây
    conditional_removal.setKeepOrganized(true);

    // Áp dụng bộ lọc ConditionalRemoval
    conditional_removal.filter(*cloud_out);

    // Xuất kết quả qua topic conditionalOR_pub_
    conditionalOR_pub_.publish(convertToPointCloud2(cloud_out));
  }

  void radiusOutlierRemoval(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_out)
  {
    // Tạo một đối tượng RadiusOutlierRemoval filter
    pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> radius_removal;

    // Thiết lập đầu vào cho bộ lọc RadiusOutlierRemoval
    radius_removal.setInputCloud(cloud_in);

    // Thiết lập bán kính tìm kiếm
    radius_removal.setRadiusSearch(radius_search_);

    // Thiết lập số lượng hàng xóm tối thiểu trong phạm vi bán kính
    radius_removal.setMinNeighborsInRadius(min_neighbors_in_radius_);

    // Áp dụng bộ lọc RadiusOutlierRemoval
    radius_removal.filter(*cloud_out);

    // Xuất kết quả qua topic radiusOR_pub_
    radiusOR_pub_.publish(convertToPointCloud2(cloud_out));
  }

  void segmentGround(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_out)
  {
    // Tạo một đối tượng SACSegmentation để phân đoạn mặt phẳng đất
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;

    // Thiết lập các tham số phân đoạn
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(ground_seg_max_iterations_);
    seg.setDistanceThreshold(ground_seg_distance_threshold_);

    // Khai báo đối tượng ModelCoefficients và PointIndices để lưu trữ kết quả phân đoạn
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr ground_indices(new pcl::PointIndices);

    // Thiết lập đầu vào cho phân đoạn
    seg.setInputCloud(cloud_in);
    seg.segment(*ground_indices, *coefficients);

    // Kiểm tra xem có điểm đất được phân đoạn hay không
    if (!ground_indices->indices.empty())
    {
      // Trích xuất các điểm không thuộc đất
      pcl::ExtractIndices<pcl::PointXYZRGB> extract;
      extract.setInputCloud(cloud_in);
      extract.setIndices(ground_indices);
      extract.setNegative(true);
      extract.filter(*cloud_out);
    }

    // Xuất kết quả qua topic obstacles_pub_
    obstacles_pub_.publish(convertToPointCloud2(cloud_out));
  }

  void euclideanClustering(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                           std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &cloud_out)
  {
    // Tạo một đối tượng KdTree để tìm kiếm hàng xóm
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(cloud_in);

    // Tạo đối tượng EuclideanClusterExtraction để phân nhóm các cụm
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_in);

    // Khai báo vector lưu trữ chỉ số của các điểm thuộc cụm
    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);

    // Tạo các điểm đám mây cho từng cụm
    int cluster_label = 0;
    for (const auto &indices : cluster_indices)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
      for (const auto &index : indices.indices)
        cluster->points.push_back(cloud_in->points[index]);
      cluster->width = cluster->points.size();
      cluster->height = 1;
      cluster->is_dense = true;
      cloud_out.push_back(cluster);
    }

    // Xuất kết quả qua topic clustered_pub_
    clustered_pub_.publish(convertToPointCloud2(cloud_out[0]));
  }

  void computeSafetyZone(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr &safety_warn_cloud,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr &safety_protect_cloud)
  {
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(cloud_in);

    // Vùng cảnh báo
    pass.setFilterFieldName("x");
    pass.setFilterLimits(safety_warn_position_(0) - safety_warn_size_ / 2,
                         safety_warn_position_(0) + safety_warn_size_ / 2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_warn_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pass.filter(*filtered_warn_cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(safety_warn_position_(1) - safety_warn_size_ / 2,
                         safety_warn_position_(1) + safety_warn_size_ / 2);
    pass.filter(*safety_warn_cloud);

    // Vùng bảo vệ
    pass.setFilterFieldName("x");
    pass.setFilterLimits(safety_protect_position_(0) - safety_protect_size_ / 2,
                         safety_protect_position_(0) + safety_protect_size_ / 2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_protect_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
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

  float calculateDistance(const pcl::PointXYZRGB &point)
  {
    float dx = point.x;
    float dy = point.y;
    float dz = point.z;

    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  std::string computeSafetyStatus(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr &safety_warn_cloud,
                                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr &safety_protect_cloud)
  {
    int num_warn = safety_warn_cloud->size();
    int num_protect = safety_protect_cloud->size();

    std::string status = "Safe";

    if (num_warn >= min_cluster_warn_size_)
    {
      consecutive_warn_count++;
      if (consecutive_warn_count >= min_consecutive_warn_count_)
      {
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
      if (consecutive_protect_count >= min_consecutive_protect_count_)
      {
        status = "Protected";
      }
    }
    else
    {
      consecutive_protect_count = 0;
    }

    return status;
  }
  void computeDistanceZ(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &safety_warn_cloud,
                      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &safety_protect_cloud)
{
  pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
  kdtree.setInputCloud(safety_warn_cloud);

  std_msgs::Float32 distance_msg;

  pcl::PointXYZRGB search_point;
  search_point.x = 0.0;
  search_point.y = 0.0;
  search_point.z = 0.0;

  std::vector<int> indices(1);
  std::vector<float> squared_distances(1);

  kdtree.nearestKSearch(search_point, 1, indices, squared_distances);

  float distance_warn = std::sqrt(squared_distances[0]);

  if (safety_protect_cloud->empty())
  {
    distance_msg.data = distance_warn;
  }
  else
  {
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree_protect;
    kdtree_protect.setInputCloud(safety_protect_cloud);

    kdtree_protect.nearestKSearch(search_point, 1, indices, squared_distances);

    float distance_protect = std::sqrt(squared_distances[0]);

    if (distance_protect < distance_warn)
    {
      distance_msg.data = distance_protect;
    }
    else
    {
      distance_msg.data = distance_warn;
    }
  }

  distance_pub_.publish(distance_msg);
}


  sensor_msgs::PointCloud2 convertToPointCloud2(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
  {
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = "base_camera";
    return cloud_msg;
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Publisher raw_pub_;
  ros::Publisher filtered_pub_;
  ros::Publisher downsampled_pub_;
  ros::Publisher denoised_pub_;
  ros::Publisher obstacles_pub_;
  ros::Publisher clustered_pub_;
  ros::Publisher output_pub_;
  ros::Publisher safety_warn_pub_;
  ros::Publisher safety_protect_pub_;
  ros::Publisher safety_status_pub_;
  ros::Publisher conditionalOR_pub_;
  ros::Publisher radiusOR_pub_;
  ros::Publisher distance_pub_;

  double pass_through_x_min_;
  double pass_through_x_max_;
  double pass_through_y_min_;
  double pass_through_y_max_;
  double pass_through_z_min_;
  double pass_through_z_max_;

  double voxel_leaf_size_;

  double condition_min_z_;
  double condition_max_z_;
  double radius_search_;
  int min_neighbors_in_radius_;

  int ground_seg_max_iterations_;
  double ground_seg_distance_threshold_;

  double cluster_tolerance_;
  int min_cluster_size_;
  int max_cluster_size_;

  double safety_warn_size_;
  double safety_protect_size_;
  Eigen::Vector3d safety_warn_position_;
  Eigen::Vector3d safety_protect_position_;

  int min_cluster_warn_size_;
  int min_cluster_protect_size_;
  int min_consecutive_warn_count_;
  int min_consecutive_protect_count_;
  int consecutive_warn_count = 0;
  int consecutive_protect_count = 0;

  void displayParameters()
  {
    ROS_INFO("Obstacle Detection Parameters:");
    ROS_INFO("pass_through_x_min: %f", pass_through_x_min_);
    ROS_INFO("pass_through_x_max: %f", pass_through_x_max_);
    ROS_INFO("pass_through_y_min: %f", pass_through_y_min_);
    ROS_INFO("pass_through_y_max: %f", pass_through_y_max_);
    ROS_INFO("pass_through_z_min: %f", pass_through_z_min_);
    ROS_INFO("pass_through_z_max: %f", pass_through_z_max_);

    ROS_INFO("voxel_leaf_size: %f", voxel_leaf_size_);

    ROS_INFO("ground_seg_max_iterations: %i", ground_seg_max_iterations_);
    ROS_INFO("ground_seg_distance_threshold: %f", ground_seg_distance_threshold_);

    ROS_INFO("cluster_tolerance: %f", cluster_tolerance_);
    ROS_INFO("min_cluster_size: %d", min_cluster_size_);
    ROS_INFO("max_cluster_size: %d", max_cluster_size_);

    ROS_INFO("safety_warn_size: %f", safety_warn_size_);
    ROS_INFO("safety_protect_size: %f", safety_protect_size_);
    ROS_INFO("safety_warn_position: [%f, %f, %f]", safety_warn_position_(0), safety_warn_position_(1), safety_warn_position_(2));
    ROS_INFO("safety_protect_position: [%f, %f, %f]", safety_protect_position_(0), safety_protect_position_(1), safety_protect_position_(2));

    ROS_INFO("min_cluster_warn_size: %i", min_cluster_warn_size_);
    ROS_INFO("min_cluster_protect_size: %i", min_cluster_protect_size_);
    ROS_INFO("min_consecutive_warn_count: %i", min_consecutive_warn_count_);
    ROS_INFO("min_consecutive_protect_count: %i", min_consecutive_protect_count_);


  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "obstacle_detection");
  ObstacleDetection obstacle_detection;
  ros::spin();
  return 0;
}
