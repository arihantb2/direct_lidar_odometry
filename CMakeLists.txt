#############################################################
#                                                           #
# Copyright (c) 2021, University of California, Los Angeles #
#                                                           #
# Authors: Kenny J. Chen, Brett T. Lopez                    #
# Contact: kennyjchen@ucla.edu, btlopez@ucla.edu            #
#                                                           #
#############################################################

cmake_minimum_required(VERSION 3.10.0)
project(direct_lidar_odometry)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

find_package( PCL REQUIRED )
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

find_package( Eigen3 REQUIRED )
include_directories(${EIGEN3_INCLUDE_DIR})

find_package( GTSAM REQUIRED )
include_directories(${GTSAM_INCLUDE_DIR})
add_definitions(${GTSAM_DEFINITIONS})
link_directories(${GTSAM_LIBRARY_DIRS})

find_package( Boost REQUIRED COMPONENTS system filesystem timer)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
  message("ERROR: OpenMP could not be found.")
endif(OPENMP_FOUND)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

set(CATKIN_PACKAGE_DEPS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  pcl_ros
  pcl_conversions
  tf_conversions
  eigen_conversions
  er_nav_msgs
  er_file_io_msgs
  er_file_io
)

find_package(catkin REQUIRED COMPONENTS ${CATKIN_PACKAGE_DEPS})

catkin_package(
  CATKIN_DEPENDS
    ${CATKIN_PACKAGE_DEPS}
  # INCLUDE_DIRS
  #   include
  LIBRARIES
    nano_gicp
    nanoflann
    frames
)

include_directories(include ${catkin_INCLUDE_DIRS})

# NanoFLANN
add_library(nanoflann STATIC
  src/nano_gicp/nanoflann.cc
)
target_link_libraries(nanoflann ${PCL_LIBRARIES})
target_include_directories(nanoflann PUBLIC include ${PCL_INCLUDE_DIRS})

# NanoGICP
add_library(nano_gicp STATIC
  src/nano_gicp/lsq_registration.cc
  src/nano_gicp/nano_gicp.cc
)
target_link_libraries(nano_gicp ${PCL_LIBRARIES} ${OpenMP_LIBS} nanoflann)
target_include_directories(nano_gicp PUBLIC include ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

# Frames
add_library(frames STATIC
  src/frames/frames.cc
  include/frames/map_io.h
)
add_dependencies(frames ${catkin_EXPORTED_TARGETS})
target_link_libraries(frames ${PCL_LIBRARIES} nano_gicp)
target_include_directories(frames PUBLIC include ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

# Odometry Node
add_executable(dlo_odom_node src/dlo/odom_node.cc src/dlo/odom.cc)
add_dependencies(dlo_odom_node ${catkin_EXPORTED_TARGETS})
target_compile_options(dlo_odom_node PRIVATE ${OpenMP_FLAGS})
target_link_libraries(dlo_odom_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${OpenMP_LIBS} Threads::Threads nano_gicp gtsam frames yaml-cpp)

# # Mapping Node
# add_executable(dlo_map_node src/dlo/map_node.cc src/dlo/map.cc)
# add_dependencies(dlo_map_node ${catkin_EXPORTED_TARGETS})
# target_compile_options(dlo_map_node PRIVATE ${OpenMP_FLAGS})
# target_link_libraries(dlo_map_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads)

# Tests
add_executable(test_nano_gicp src/test/test_nano_gicp.cpp)
add_dependencies(test_nano_gicp ${catkin_EXPORTED_TARGETS})
target_compile_options(test_nano_gicp PRIVATE ${OpenMP_FLAGS})
target_link_libraries(test_nano_gicp ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads nano_gicp)

add_executable(test_transform_point src/test/test_transform_point.cpp)
add_dependencies(test_transform_point ${catkin_EXPORTED_TARGETS})
target_link_libraries(test_transform_point ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(test_quaternion_angles src/test/test_quaternion_angles.cpp)
add_dependencies(test_quaternion_angles ${catkin_EXPORTED_TARGETS})
target_link_libraries(test_quaternion_angles ${catkin_LIBRARIES})

include(CTest)
add_test(NAME nano_gicp COMMAND $<TARGET_FILE:test_nano_gicp>)
add_test(NAME transform_point COMMAND $<TARGET_FILE:test_transform_point>)
add_test(NAME quaternion_angles COMMAND $<TARGET_FILE:test_quaternion_angles>)

## Install
install(
  TARGETS
    # Libraries
    nano_gicp
    nanoflann
    frames

    # Executables
    dlo_odom_node
    # dlo_map_node

  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
    cfg
    launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
