find_package(Ceres REQUIRED)





add_library(toyslam_bundleadjustment_library

)

target_link_libraries(toyslam_bundleadjustment_library
    Ceres::ceres
    
    glog
)