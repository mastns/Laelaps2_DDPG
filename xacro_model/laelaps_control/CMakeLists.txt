cmake_minimum_required(VERSION 2.8.3)
project(laelaps_control)

find_package(catkin REQUIRED COMPONENTS roscpp actionlib rospy std_msgs message_generation)
find_package(gazebo REQUIRED)

add_message_files(
  FILES
  Toe.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)
include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

add_executable(toe4_ellipse_publisher src/toe4_ellipse_publisher.cpp)
target_link_libraries(toe4_ellipse_publisher
  ${GAZEBO_LIBRARIES}
  pthread
  ${catkin_LIBRARIES}
)
add_dependencies(toe4_ellipse_publisher toe4_ellipse_publisher_generate_messages_cpp)