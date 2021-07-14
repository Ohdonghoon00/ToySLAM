add_library(toyslam_visualization_library
  image_viewer.cpp
  map_viewer.cpp
)

target_link_libraries(toyslam_visualization_library

    GL
    GLU
    glut
)