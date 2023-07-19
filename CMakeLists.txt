cmake_minimum_required(VERSION 3.0.2)
project(obstacle_detection)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  tf2
  tf2_ros
  message_generation
  std_msgs
)
## Generate messages in the 'msg' folder
add_message_files(
  FILES
  fields_safety.msg
)
generate_messages(
  DEPENDENCIES
  std_msgs
)
catkin_package()


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${pcl_ros_INCLUDE_DIRS}
  ${tf2_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}

)

add_executable(obstacle_detection src/obstacle_detection.cpp)
target_link_libraries(obstacle_detection ${catkin_LIBRARIES})
