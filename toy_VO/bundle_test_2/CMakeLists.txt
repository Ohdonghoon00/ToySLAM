cmake_minimum_required(VERSION 3.5.0)
project(Toy_VO VERSION 0.1.0)
set(CMAKE_BUILD_TYPE Release)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Ceres_INCLUDE_DIRS})

set(SRC_DIR	"${CMAKE_SOURCE_DIR}")

# file(GLOB APP_SOURCES "${SRC_DIR}/*.cpp")
# foreach(app_source ${APP_SOURCES})
#     string(REPLACE ".cpp" "" app_name ${app_source})
#     string(REPLACE "${SRC_DIR}/" "" app_name ${app_name})    
#     add_executable(${app_name} ${app_source})
#     add_library(${app_name} math.cpp)
#     target_link_libraries(${app_name} ${OpenCV_LIBS})
#     # target_link_libraries(${app_name} Ceres::ceres)
#     target_link_libraries(${app_name} ${Ceres_LIBS})
#     target_link_libraries(${app_name} GL)
#     target_link_libraries(${app_name} GLU)
#     target_link_libraries(${app_name} glut)
# endforeach(app_source ${APP_SOURCES})


add_executable(5local_ba_test 
    5local_ba_test.cpp
    math.cpp
)
target_link_libraries(5local_ba_test 
    ${OpenCV_LIBS}
    GL
    GLU
    glut
)

add_executable(final_BA 
    final_BA.cpp
    math.cpp    
)
target_link_libraries(final_BA 
    ${OpenCV_LIBS}
    GL
    GLU
    glut
)





