#pragma once


#include <string>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <sstream>





std::string Read_Kitti_image_Data(char **kitti_data_num);


double* Read_Kitti_Calibration_Data(char **kitti_calibration_data_num, double intrinsic_[]);

std::ifstream Read_Kiiti_GT_Data(char **kitti_gt_data_num);