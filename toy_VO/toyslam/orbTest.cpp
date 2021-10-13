// #include "assembly.h"
#include "types/Map.h"
#include "types/Frame.h"
#include "types/Data.h"
#include "types/common.h"
#include "types/definition.h"

#include "visualization/map_viewer.h"



#include "opencv2/opencv.hpp"
#include "GL/freeglut.h" 
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unistd.h>
#include <cmath>
#include "opencv2/sfm/triangulation.hpp"
#include "opencv2/sfm/projection.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>

// #include "ceres/ceres.h"
// #include "gurobi_c++.h"


// using namespace std;
// using namespace cv;
// using namespace DBoW2;
// using namespace g2o;
// using ceres::CauchyLoss;
// using ceres::HuberLoss;


int main(int argc, char **argv)
{       

//////////////////////////////////////// Read data ///////////////////////////////////
    
    // Read Calibration Data
    double intrinsic[12];  
    Read_Kitti_Calibration_Data(argv, intrinsic);
    
    // f : focal length , c : principal point (c.x, c.y) , K : Camera Matrix
    double f = intrinsic[0];
    cv::Point2d c(intrinsic[2], intrinsic[6]);
    const cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, c.x, 0, f, c.y, 0, 0, 1);
    std::vector<double> dist_coeff = { -0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194 };
            
    std::cout << " Camera Matrix  : " << std::endl << K << std::endl;
    
    
    // Read Image Data
    cv::VideoCapture video;
    if (!video.open(Read_Kitti_image_Data(argv))) return -1;
    
    // Read GT Data
    std::ifstream GT_DATA = Read_Kiiti_GT_Data(argv);

    
    glutInit(&argc, argv);
    initialize_window();  

/////////////////////////////////////////////////////////////////////////////////////////////
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat mask, inlier_mask; 

/////////////////////////////////////////////////////////////////////////////////////////////
    
    while(true)
    {

    
        // Img Load
        Frame PrevImg(0);
        video >> PrevImg.frame;
        if(PrevImg.frame.empty()) cv::waitKey();
        if(PrevImg.frame.channels() > 1) cv::cvtColor(PrevImg.frame, PrevImg.frame, cv::COLOR_RGB2GRAY);

        // Extract Keypoint and Descriptor
        cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
        orb->detectAndCompute(PrevImg.frame, mask, keypoints, descriptors);



        // Feature Matching
        


        // Show Image
        cv::Mat KeypointImg;
        cv::drawKeypoints(PrevImg.frame, keypoints, KeypointImg, cv::Scalar(0, 0, 255));
        imshow("KeypointImg", KeypointImg);
        cv::waitKey();

    }
    return 0;
}