cmake_minimum_required(VERSION 3.5.0)
project(Toy_VO VERSION 0.1.0)
set(CMAKE_BUILD_TYPE Release)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Ceres_INCLUDE_DIRS})

# set(SRC_DIR	"${CMAKE_SOURCE_DIR}")

option(BUILD_DBoW2   "Build DBoW2"            ON)


# add_executable(5local_ba_test 
#     5local_ba_test.cpp
#     math.cpp
# )
# target_link_libraries(5local_ba_test 
#     ${OpenCV_LIBS}
#     GL
#     GLU
#     glut
# )

# set(HDRS
#   DBoW2/include/DBoW2/BowVector.h           DBoW2/include/DBoW2/FBrief.h
#   DBoW2/include/DBoW2/QueryResults.h        DBoW2/include/DBoW2/TemplatedDatabase.h   DBoW2/include/DBoW2/FORB.h
#   DBoW2/include/DBoW2/DBoW2.h               DBoW2/include/DBoW2/FClass.h              DBoW2/include/DBoW2/FeatureVector.h
#   DBoW2/include/DBoW2/ScoringObject.h       DBoW2/include/DBoW2/TemplatedVocabulary.h)
set(SRCS 
  DBoW2/src/BowVector.cpp     DBoW2/src/FBrief.cpp        DBoW2/src/FORB.cpp
  DBoW2/src/FeatureVector.cpp DBoW2/src/QueryResults.cpp  DBoW2/src/ScoringObject.cpp)
# set(DEPENDENCY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
# set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)
if(BUILD_DBoW2)
  set(LIB_SHARED "SHARED")
  if(WIN32)
    set(LIB_SHARED "STATIC")
  endif(WIN32)
  add_library(DBoW2 ${LIB_SHARED} ${SRCS})
  target_include_directories(DBoW2 PUBLIC DBoW2/include/DBoW2/ DBoW2/include/)
  target_link_libraries(DBoW2 ${OpenCV_LIBS})
  set_target_properties(DBoW2 PROPERTIES CXX_STANDARD 11)
endif(BUILD_DBoW2)



add_executable(loop_detection_test 
    loop_detection_test.cpp
    math.cpp
    g2o_test.cpp    
)
# add_library(loop_detection_test ${LIB_SHARED} ${SRCS})
# target_include_directories(loop_detection_test PUBLIC DBoW2/include/DBoW2/ DBoW2/include/)
target_link_libraries(loop_detection_test DBoW2
    ${OpenCV_LIBS}
    # ${SRCS}
    ${CSPARSE_LIBRARY}
    GL
    GLU
    glut
    g2o_core g2o_stuff g2o_types_sba g2o_types_slam3d 
    g2o_csparse_extension g2o_solver_csparse
    cholmod g2o_solver_cholmod
)





