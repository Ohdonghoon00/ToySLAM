cmake_minimum_required(VERSION 3.5.0)
project(toyslam)
set(CMAKE_BUILD_TYPE Release)
find_package(OpenCV REQUIRED)
find_package(Eigen)





include_directories(
  
  "/usr/include/eigen3"
  ${OpenCV_INCLUDE_DIRS}
)






add_executable(toyslam_main
    toyslam_main.cpp
    math.cpp
    
)
add_subdirectory(types)
add_subdirectory(feature)
add_subdirectory(track)
add_subdirectory(bundleAdjustment)
add_subdirectory(graph_optimization)
add_subdirectory(visualization)

# add_library(toyslam toyslam.cpp)
target_link_libraries(toyslam_main

  ${OpenCV_LIBS}
  toyslam_types_library
  toyslam_feature_library
  toyslam_track_library
  toyslam_bundleadjustment_library
  toyslam_graph_optimization_library
  toyslam_visualization_library
)

add_executable(Eigen_test
    
    Eigen_test.cpp
    
)