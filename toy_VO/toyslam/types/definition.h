#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unistd.h>
#include <cmath>


// definition
typedef Eigen::Matrix <bool, Eigen::Dynamic, 1> VectorXb;
// Eigen::Matrix<bool, Eigen::Dynamic , Eigen::Dynamic> MatrixXb

// Keyframe Selection
int KS_track_overlap_ratio= 65;
int KS_inliers_num = 300;
double KS_yaw_difference = 100; // 0.055

