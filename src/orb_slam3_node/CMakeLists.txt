cmake_minimum_required(VERSION 3.8)
project(orb_slam3_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

# Point directly to your compiled core directory
set(ORB_SLAM3_DIR /home/diksha/slam_ws/src/ORB_SLAM3)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  include
  ${ORB_SLAM3_DIR}
  ${ORB_SLAM3_DIR}/include
  ${ORB_SLAM3_DIR}/include/CameraModels
  ${ORB_SLAM3_DIR}/Thirdparty/Sophus
)

add_executable(orb_slam3_node src/orb_slam3_node.cpp)

target_link_libraries(orb_slam3_node
  ${OpenCV_LIBS}
  ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so
  ${Pangolin_LIBRARIES}
)

ament_target_dependencies(orb_slam3_node
  rclcpp
  sensor_msgs
  geometry_msgs
  message_filters
  cv_bridge
)

install(TARGETS
  orb_slam3_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()