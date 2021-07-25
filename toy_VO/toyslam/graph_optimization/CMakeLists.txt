
find_package(g2o REQUIRED)


option(BUILD_DBoW2   "Build DBoW2"            ON)

set(HDRS
  DBoW2/include/DBoW2/BowVector.h           DBoW2/include/DBoW2/FBrief.h
  DBoW2/include/DBoW2/QueryResults.h        DBoW2/include/DBoW2/TemplatedDatabase.h   DBoW2/include/DBoW2/FORB.h
  DBoW2/include/DBoW2/DBoW2.h               DBoW2/include/DBoW2/FClass.h              DBoW2/include/DBoW2/FeatureVector.h
  DBoW2/include/DBoW2/ScoringObject.h       DBoW2/include/DBoW2/TemplatedVocabulary.h)

  set(SRCS 
  DBoW2/src/BowVector.cpp     DBoW2/src/FBrief.cpp        DBoW2/src/FORB.cpp
  DBoW2/src/FeatureVector.cpp DBoW2/src/QueryResults.cpp  DBoW2/src/ScoringObject.cpp)

set(G2O_LIBS
  g2o_core 
  g2o_types_sba 
  g2o_types_slam3d
  g2o_solver_csparse 
  g2o_stuff 
  g2o_csparse_extension
  g2o_solver_cholmod
  g2o_types_sim3
  cholmod
)

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

add_library(toyslam_graph_optimization_library
  global_optimize.cpp
  loop_detection.cpp
  pose_graph.cpp
)

target_link_libraries(toyslam_graph_optimization_library DBoW2
    ${CSPARSE_LIBRARY}
    ${G2O_LIBS}

)
