#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

class ObstacleDetection
{
public:
  ObstacleDetection() : nh_("~")
  {
    // Đăng ký các subscriber và publisher
    pointcloud_sub_ = nh_.subscribe("/camera/depth/color/points", 1, &ObstacleDetection::pointcloudCallback, this);
    filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("filtered_publisher", 1);
    downsampled_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("downsampled_publisher", 1);
    obstacles_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("obstacles_publisher", 1);
    clustered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("clustered_publisher", 1);
  }

  void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr obstacles_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // Chuyển đổi dữ liệu từ sensor_msgs::PointCloud2 sang pcl::PointCloud
    pcl::fromROSMsg(*cloud_msg, *input_cloud);

    // Bước 1: PassThrough filter
    passThroughFilter(input_cloud, filtered_cloud);

    // Bước 2: VoxelGrid filter
    voxelGridFilter(filtered_cloud, downsampled_cloud);

    // Bước 3: RANSAC segmentation
    ransacSegmentation(downsampled_cloud, obstacles_cloud);

    // Bước 4: Loại bỏ sàn nhà
    removeFloor(obstacles_cloud, obstacles_cloud);

    // Bước 5: Euclidean clustering
    euclideanClustering(obstacles_cloud, clustered_cloud);

    // Xuất dữ liệu qua các topic tương ứng
    filtered_pub_.publish(convertToROSMsg(filtered_cloud));
    downsampled_pub_.publish(convertToROSMsg(downsampled_cloud));
    obstacles_pub_.publish(convertToROSMsg(obstacles_cloud));
    clustered_pub_.publish(convertToROSMsg(clustered_cloud));
  }

  void passThroughFilter(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr& filtered_cloud)
  {
    // TODO: Cài đặt PassThrough filter để lọc điểm dữ liệu trong một khoảng giá trị cho trước
    // Sử dụng filtered_cloud để lưu trữ kết quả
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(input_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 1.0);  // Giới hạn khoảng giá trị z từ 0.0 đến 1.0
    pass.filter(*filtered_cloud);
  }

  void voxelGridFilter(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr& downsampled_cloud)
  {
    // TODO: Cài đặt VoxelGrid filter để giảm độ phân giải của điểm dữ liệu
    // Sử dụng downsampled_cloud để lưu trữ kết quả
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
    voxel_grid.setInputCloud(input_cloud);
    voxel_grid.setLeafSize(0.01, 0.01, 0.01);  // Độ phân giải voxel là 0.01
    voxel_grid.filter(*downsampled_cloud);
  }

  void ransacSegmentation(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr& segmented_cloud)
  {
    // TODO: Cài đặt RANSAC segmentation để phân đoạn điểm dữ liệu thành các mặt phẳng
    // Sử dụng segmented_cloud để lưu trữ kết quả
   pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.02);  // Đặt ngưỡng khoảng cách để tạo mặt phẳng
    seg.setInputCloud(input_cloud);
    seg.segment(*inliers, *coefficients);

    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*segmented_cloud);
  }

  void removeFloor(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr& filtered_cloud)
  {
    // TODO: Triển khai phần loại bỏ sàn nhà từ input_cloud và lưu kết quả vào filtered_cloud
    // Gợi ý: Sử dụng RANSAC segmentation tương tự như trong hàm ransacSegmentation()
    // Chú ý đến các thông số như ngưỡng khoảng cách, độ phân giải, v.v.
 // Tính toán trung bình của các điểm dữ liệu theo trục z
  float z_mean = 0.0;
  for (const auto& point : input_cloud->points) {
    z_mean += point.z;
  }
  z_mean /= input_cloud->size();

  // Lọc các điểm dữ liệu có trục z dưới ngưỡng trung bình
  filtered_cloud->clear();
  for (const auto& point : input_cloud->points) {
    if (point.z > z_mean) {
      filtered_cloud->push_back(point);
    }
  }
  }

  void euclideanClustering(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr& clustered_cloud)
  {
    // TODO: Triển khai phần Euclidean clustering từ input_cloud và lưu kết quả vào clustered_cloud
    // Gợi ý: Sử dụng pcl::EuclideanClusterExtraction và pcl::search::KdTree
    // Chú ý đến các thông số như khoảng cách chấp nhận được, số điểm tối thiểu trong mỗi cụm, v.v.
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud(input_cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance(0.1);  // Khoảng cách chấp nhận được giữa các điểm trong cụm
  ec.setMinClusterSize(100);    // Số điểm tối thiểu trong mỗi cụm
  ec.setMaxClusterSize(25000);  // Số điểm tối đa trong mỗi cụm
  ec.setSearchMethod(tree);
  ec.setInputCloud(input_cloud);
  ec.extract(cluster_indices);

  clustered_cloud->clear();
  for (const auto& indices : cluster_indices) {
    for (const auto& index : indices.indices) {
      clustered_cloud->push_back(input_cloud->points[index]);
    }
  }
  }

  sensor_msgs::PointCloud2 convertToROSMsg(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)
  {
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header.frame_id = "camera_link";  // Thay đổi frame_id tùy thuộc vào hệ tọa độ của camera
    return ros_cloud;
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber pointcloud_sub_;
  ros::Publisher filtered_pub_;
  ros::Publisher downsampled_pub_;
  ros::Publisher obstacles_pub_;
  ros::Publisher clustered_pub_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "obstacle_detection");
  ObstacleDetection obstacle_detection;
  ros::spin();
  return 0;
}
