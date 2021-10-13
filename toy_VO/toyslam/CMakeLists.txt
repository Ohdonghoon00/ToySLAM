cmake_minimum_required(VERSION 3.5.0)
project(toyslam)
set(CMAKE_BUILD_TYPE Release)
find_package(OpenCV REQUIRED )
find_package(Ceres REQUIRED)

enable_language(CXX)
set(EXTRA_LIBS /opt/gurobi912/linux64/lib/libgurobi91.so
/opt/gurobi912/linux64/lib/libgurobi_c++.a
/opt/gurobi912/linux64/lib/libgurobi.so.9.1.2)



include_directories(
  "/opt/gurobi912/linux64/include"
  "/usr/include/eigen3"
  ${OpenCV_INCLUDE_DIRS}
)






add_executable(toyslam_main
toy_slam_ORBmap2.cpp
    gurobi_helper.cpp
)



add_subdirectory(types)
add_subdirectory(feature)
add_subdirectory(track)
add_subdirectory(graph_optimization)
add_subdirectory(visualization)

target_link_libraries(toyslam_main
  
  ${OpenCV_LIBS}
  toyslam_types_library
  toyslam_feature_library
  toyslam_track_library
  toyslam_graph_optimization_library
  toyslam_visualization_library
  ${EXTRA_LIBS}
  Ceres::ceres
  glog
    
)

add_executable(orbTest 
orbTest.cpp)




target_link_libraries(orbTest
  
  ${OpenCV_LIBS}
  toyslam_types_library
  # toyslam_feature_library
  # toyslam_track_library
  # toyslam_graph_optimization_library
  toyslam_visualization_library
  # ${ORB_TEST_LIBS}
  # Ceres::ceres
  # glog
    
)

add_executable(gurobi_test 
gurobi_test.cpp)


    
target_link_libraries(gurobi_test ${EXTRA_LIBS} )

