cmake_minimum_required(VERSION 2.8.3)
project(bounding_box_server)

## Compile as C++14, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE=Release)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  tf
)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

## Generate messages in the 'msg' folder
 add_message_files(
   FILES
   BoundingBox.msg
   BoundingBoxes.msg
 )

## Generate services in the 'srv' folder
 add_service_files(
   FILES
   GetBoundingBox.srv
 )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
#   actionlib_msgs
 )

catkin_package(
  
)
###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

## Give all the sources needed to compile 
set(SOURCE_CODE 
    src/bounding_box_server_node.cpp 
    src/bounding_box_server.cpp
    src/bounding_box_server_common.cpp 
    src/bounding_box_server_filters.cpp
    src/bounding_box_publisher.cpp
    src/bounding_box_server_transform.cpp
    src/bounding_box_server_service.cpp
)

## Create the execute using the source files specified
add_executable(${PROJECT_NAME}_node ${SOURCE_CODE})

## Generate messages before compiling (useful for compiling empty workspace)
add_dependencies(${PROJECT_NAME}_node bounding_box_server_generate_messages_cpp)

target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
)
