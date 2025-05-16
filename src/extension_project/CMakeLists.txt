cmake_minimum_required(VERSION 3.0.2)
project(extension_project)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  intera_interface
  roscpp
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  gazebo_ros
)

## Declare catkin package
catkin_package(
  CATKIN_DEPENDS
    intera_interface
    roscpp
    rospy
    sensor_msgs
    std_msgs
    cv_bridge
    gazebo_ros
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Mark executable scripts (Python etc.) for installation
catkin_install_python(PROGRAMS
  scripts/camera.py
  scripts/camera_direct.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other scripts for installation
install(PROGRAMS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../kill_gazebo.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files, config files, etc.
install(DIRECTORY
  launch
  config
  urdf
  model
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
